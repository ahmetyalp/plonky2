use std::collections::HashMap;
use std::time::Instant;

use log::info;
use rayon::prelude::*;

use crate::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::field::extension_field::Extendable;
use crate::field::field::Field;
use crate::generator::generate_partial_witness;
use crate::plonk_challenger::Challenger;
use crate::plonk_common::{PlonkPolynomials, ZeroPolyOnCoset};
use crate::polynomial::commitment::ListPolynomialCommitment;
use crate::polynomial::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::proof::Proof;
use crate::timed;
use crate::util::marking::MarkedTargets;
use crate::util::partial_products::partial_products;
use crate::util::{log2_ceil, transpose};
use crate::vanishing_poly::{
    eval_vanishing_poly_base, evaluate_gate_constraints, evaluate_gate_constraints_base, yoba,
};
use crate::vars::{EvaluationVars, EvaluationVarsBase};
use crate::witness::{PartialWitness, Witness};

pub(crate) fn prove<F: Extendable<D>, const D: usize>(
    prover_data: &ProverOnlyCircuitData<F, D>,
    common_data: &CommonCircuitData<F, D>,
    inputs: PartialWitness<F>,
    marked: Vec<MarkedTargets>,
) -> Proof<F, D> {
    let fri_config = &common_data.config.fri_config;
    let config = &common_data.config;
    let num_wires = config.num_wires;
    let num_challenges = config.num_challenges;
    let quotient_degree = common_data.quotient_degree();
    let degree = common_data.degree();

    let start_proof_gen = Instant::now();

    let mut partial_witness = inputs;
    info!("Running {} generators", prover_data.generators.len());
    timed!(
        generate_partial_witness(&mut partial_witness, &prover_data.generators),
        "to generate witness"
    );

    let witness = timed!(
        partial_witness.full_witness(degree, num_wires),
        "to compute full witness"
    );

    for m in marked {
        m.display(&witness);
    }
    timed!(
        witness
            .check_copy_constraints(&prover_data.copy_constraints, &prover_data.gate_instances)
            .unwrap(), // TODO: Change return value to `Result` and use `?` here.
        "to check copy constraints"
    );

    let wires_values: Vec<PolynomialValues<F>> = timed!(
        witness
            .wire_values
            .iter()
            .map(|column| PolynomialValues::new(column.clone()))
            .collect(),
        "to compute wire polynomials"
    );

    // TODO: Could try parallelizing the transpose, or not doing it explicitly, instead having
    // merkle_root_bit_rev_order do it implicitly.
    let wires_commitment = timed!(
        ListPolynomialCommitment::new(
            wires_values,
            fri_config.rate_bits,
            PlonkPolynomials::WIRES.blinding
        ),
        "to compute wires commitment"
    );

    let mut challenger = Challenger::new();
    // Observe the instance.
    // TODO: Need to include public inputs as well.
    challenger.observe_hash(&common_data.circuit_digest);

    challenger.observe_hash(&wires_commitment.merkle_tree.root);
    let betas = challenger.get_n_challenges(num_challenges);
    let gammas = challenger.get_n_challenges(num_challenges);

    assert!(
        common_data.quotient_degree_factor + 1 <=common_data.config.num_routed_wires,
        "When the number of routed wires is smaller that the degree, we should change the logic to avoid computing partial products."
    );
    let mut partial_products = timed!(
        all_wires_permutation_partial_products(&witness, &betas, &gammas, prover_data, common_data),
        "to compute partial products"
    );

    let plonk_z_vecs = timed!(compute_zs(&partial_products, common_data), "to compute Z's");

    // The first polynomial in `partial_products` represent the final product used in the
    // computation of `Z`. It isn't needed anymore so we discard it.
    partial_products.iter_mut().for_each(|part| {
        part.remove(0);
    });

    let zs_partial_products = [plonk_z_vecs, partial_products.concat()].concat();
    let zs_partial_products_commitment = timed!(
        ListPolynomialCommitment::new(
            zs_partial_products,
            fri_config.rate_bits,
            PlonkPolynomials::ZS_PARTIAL_PRODUCTS.blinding
        ),
        "to commit to Z's"
    );

    challenger.observe_hash(&zs_partial_products_commitment.merkle_tree.root);

    let alphas = challenger.get_n_challenges(num_challenges);

    {
        let get_at_index = |comm: &ListPolynomialCommitment<F>, i: usize| -> Vec<F> {
            comm.original_values
                .iter()
                .map(|v| v.values[i])
                .collect::<Vec<_>>()
        };
        let mut nums = HashMap::<F, usize>::new();
        let mut dens = HashMap::<F, usize>::new();
        let points = F::two_adic_subgroup(common_data.degree_bits);
        for i in 0..degree {
            let x = points[i];
            let local_constants_sigmas = get_at_index(&prover_data.constants_sigmas_commitment, i);
            let local_constants = &local_constants_sigmas[common_data.constants_range()];
            let s_sigmas = &local_constants_sigmas[common_data.sigmas_range()];
            let local_wires = get_at_index(&wires_commitment, i);
            let vars = EvaluationVarsBase {
                local_constants,
                local_wires: &local_wires,
            };
            let numerator_values = (0..common_data.config.num_routed_wires).for_each(|j| {
                let wire_value = vars.local_wires[j];
                let k_i = common_data.k_is[j];
                let s_id = k_i * x;
                *nums
                    .entry(wire_value + betas[0] * s_id + gammas[0])
                    .or_default() += 1;
            });
            let denominator_values = (0..common_data.config.num_routed_wires).for_each(|j| {
                let wire_value = vars.local_wires[j];
                let s_sigma = s_sigmas[j];
                *dens
                    .entry(wire_value + betas[0] * s_sigma + gammas[0])
                    .or_default() += 1;
            });
        }
        println!("yo");
        for (k, v) in nums.iter() {
            if let Some(w) = dens.get(&k) {
                if *v != *w {
                    println!("Bad: {} {} {}", *k, *v, *w);
                }
            } else {
                println!("Bad: {} {}", *k, *v);
            }
        }
        println!("ya");
        for (k, v) in dens.iter() {
            if let Some(w) = nums.get(&k) {
                if *v != *w {
                    println!("Bad: {} {} {}", *k, *v, *w);
                }
            } else {
                println!("Bad: {} {}", *k, *v);
            }
        }
        println!("yu");

        let mut bam = F::ONE;
        for (k, v) in nums.iter() {
            bam *= k.exp(*v as u64);
        }
        dbg!(bam);
        let mut boom = F::ONE;
        for (k, v) in dens.iter() {
            boom *= k.exp(*v as u64);
        }
        dbg!(boom);
    }
    {
        let get_at_index = |comm: &ListPolynomialCommitment<F>, i: usize| -> Vec<F> {
            comm.original_values
                .iter()
                .map(|v| v.values[i])
                .collect::<Vec<_>>()
        };
        let mut nums = vec![F::ONE; common_data.config.num_challenges];
        let mut dens = vec![F::ONE; common_data.config.num_challenges];
        let points = F::two_adic_subgroup(common_data.degree_bits);
        for i in 0..degree {
            let x = points[i];
            let local_constants_sigmas = get_at_index(&prover_data.constants_sigmas_commitment, i);
            let local_constants = &local_constants_sigmas[common_data.constants_range()];
            let s_sigmas = &local_constants_sigmas[common_data.sigmas_range()];
            let local_wires = get_at_index(&wires_commitment, i);
            let vars = EvaluationVarsBase {
                local_constants,
                local_wires: &local_wires,
            };
            for ii in 0..common_data.config.num_challenges {
                let numerator_values = (0..common_data.config.num_routed_wires)
                    .map(|j| {
                        let wire_value = vars.local_wires[j];
                        let k_i = common_data.k_is[j];
                        let s_id = k_i * x;
                        wire_value + betas[ii] * s_id + gammas[ii]
                    })
                    .collect::<Vec<_>>();
                let denominator_values = (0..common_data.config.num_routed_wires)
                    .map(|j| {
                        let wire_value = vars.local_wires[j];
                        let s_sigma = s_sigmas[j];
                        wire_value + betas[ii] * s_sigma + gammas[ii]
                    })
                    .collect::<Vec<_>>();

                nums[ii] *= numerator_values.into_iter().product();
                dens[ii] *= denominator_values.into_iter().product();
            }
        }
        dbg!(nums, dens);
    }
    {
        let get_at_index = |comm: &ListPolynomialCommitment<F>, i: usize| -> Vec<F> {
            comm.original_values
                .iter()
                .map(|v| v.values[i])
                .collect::<Vec<_>>()
        };
        let points = F::two_adic_subgroup(common_data.degree_bits);
        for i in 0..degree {
            let x = points[i];
            let i_next = (i + 1) % degree;
            let local_constants_sigmas = get_at_index(&prover_data.constants_sigmas_commitment, i);
            let local_constants = &local_constants_sigmas[common_data.constants_range()];
            let s_sigmas = &local_constants_sigmas[common_data.sigmas_range()];
            let local_wires = get_at_index(&wires_commitment, i);
            let local_zs_partial_products = get_at_index(&zs_partial_products_commitment, i);
            let local_zs = &local_zs_partial_products[common_data.zs_range()];
            let next_zs =
                &get_at_index(&zs_partial_products_commitment, i_next)[common_data.zs_range()];
            let partial_products = &local_zs_partial_products[common_data.partial_products_range()];

            debug_assert_eq!(local_wires.len(), common_data.config.num_wires);
            debug_assert_eq!(local_zs.len(), num_challenges);

            let vars = EvaluationVarsBase {
                local_constants,
                local_wires: &local_wires,
            };
            let mut quotient_values = yoba(
                common_data,
                i,
                x,
                vars,
                local_zs,
                next_zs,
                partial_products,
                s_sigmas,
                &betas,
                &gammas,
                &alphas,
            );
            assert!(
                quotient_values.iter().all(|yy| yy.is_zero()),
                "{}-th gate ({}) constraints not satisfied.\n {:?}",
                i,
                prover_data.gate_instances[i].gate_type.0.id(),
                quotient_values
            );
        }
    }

    let quotient_polys = timed!(
        compute_quotient_polys(
            common_data,
            prover_data,
            &wires_commitment,
            &zs_partial_products_commitment,
            &betas,
            &gammas,
            &alphas,
        ),
        "to compute vanishing polys"
    );

    // Compute the quotient polynomials, aka `t` in the Plonk paper.
    let all_quotient_poly_chunks = timed!(
        quotient_polys
            .into_par_iter()
            .flat_map(|mut quotient_poly| {
                quotient_poly.trim();
                quotient_poly.pad(quotient_degree).expect(
                    "Quotient has failed, the vanishing polynomial is not divisible by `Z_H",
                );
                // Split t into degree-n chunks.
                quotient_poly.chunks(degree)
            })
            .collect(),
        "to compute quotient polys"
    );

    let quotient_polys_commitment = timed!(
        ListPolynomialCommitment::new_from_polys(
            all_quotient_poly_chunks,
            fri_config.rate_bits,
            PlonkPolynomials::QUOTIENT.blinding
        ),
        "to commit to quotient polys"
    );

    challenger.observe_hash(&quotient_polys_commitment.merkle_tree.root);

    let zeta = challenger.get_extension_challenge();

    let (opening_proof, openings) = timed!(
        ListPolynomialCommitment::open_plonk(
            &[
                &prover_data.constants_sigmas_commitment,
                &wires_commitment,
                &zs_partial_products_commitment,
                &quotient_polys_commitment,
            ],
            zeta,
            &mut challenger,
            common_data,
        ),
        "to compute opening proofs"
    );

    info!(
        "{:.3}s for overall witness & proof generation",
        start_proof_gen.elapsed().as_secs_f32()
    );

    Proof {
        wires_root: wires_commitment.merkle_tree.root,
        plonk_zs_root: zs_partial_products_commitment.merkle_tree.root,
        quotient_polys_root: quotient_polys_commitment.merkle_tree.root,
        openings,
        opening_proof,
    }
}

/// Compute the partial products used in the `Z` polynomials.
fn all_wires_permutation_partial_products<F: Extendable<D>, const D: usize>(
    witness: &Witness<F>,
    betas: &[F],
    gammas: &[F],
    prover_data: &ProverOnlyCircuitData<F, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<Vec<PolynomialValues<F>>> {
    (0..common_data.config.num_challenges)
        .map(|i| {
            wires_permutation_partial_products(
                witness,
                betas[i],
                gammas[i],
                prover_data,
                common_data,
            )
        })
        .collect()
}

/// Compute the partial products used in the `Z` polynomial.
/// Returns the polynomials interpolating `partial_products(f / g)`
/// where `f, g` are the products in the definition of `Z`: `Z(g^i) = f / g`.
fn wires_permutation_partial_products<F: Extendable<D>, const D: usize>(
    witness: &Witness<F>,
    beta: F,
    gamma: F,
    prover_data: &ProverOnlyCircuitData<F, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.quotient_degree_factor;
    let subgroup = &prover_data.subgroup;
    let k_is = &common_data.k_is;
    let values = subgroup
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            let s_sigmas = &prover_data.sigmas[i];
            let quotient_values = (0..common_data.config.num_routed_wires)
                .map(|j| {
                    let wire_value = witness.get_wire(i, j);
                    let k_i = k_is[j];
                    let s_id = k_i * x;
                    let s_sigma = s_sigmas[j];
                    let numerator = wire_value + beta * s_id + gamma;
                    let denominator = wire_value + beta * s_sigma + gamma;
                    numerator / denominator
                })
                .collect::<Vec<_>>();

            let quotient_partials = partial_products(&quotient_values, degree);

            // This is the final product for the quotient.
            let quotient = quotient_partials
                [common_data.num_partial_products.0 - common_data.num_partial_products.1..]
                .iter()
                .copied()
                .product();

            // We add the quotient at the beginning of the vector to reuse them later in the computation of `Z`.
            [vec![quotient], quotient_partials].concat()
        })
        .collect::<Vec<_>>();

    transpose(&values)
        .into_par_iter()
        .map(PolynomialValues::new)
        .collect()
}

fn compute_zs<F: Extendable<D>, const D: usize>(
    partial_products: &[Vec<PolynomialValues<F>>],
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    (0..common_data.config.num_challenges)
        .map(|i| compute_z(&partial_products[i], common_data))
        .collect()
}

/// Compute the `Z` polynomial by reusing the computations done in `wires_permutation_partial_products`.
fn compute_z<F: Extendable<D>, const D: usize>(
    partial_products: &[PolynomialValues<F>],
    common_data: &CommonCircuitData<F, D>,
) -> PolynomialValues<F> {
    let mut plonk_z_points = vec![F::ONE];
    for i in 1..common_data.degree() {
        let quotient = partial_products[0].values[i - 1];
        let last = *plonk_z_points.last().unwrap();
        plonk_z_points.push(last * quotient);
    }
    plonk_z_points.into()
}

fn compute_quotient_polys<'a, F: Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, D>,
    wires_commitment: &'a ListPolynomialCommitment<F>,
    zs_partial_products_commitment: &'a ListPolynomialCommitment<F>,
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
) -> Vec<PolynomialCoeffs<F>> {
    let num_challenges = common_data.config.num_challenges;
    let max_degree_bits = log2_ceil(common_data.quotient_degree_factor + 1);
    assert!(
        max_degree_bits <= common_data.config.rate_bits,
        "Having constraints of degree higher than the rate is not supported yet. \
        If we need this in the future, we can precompute the larger LDE before computing the `ListPolynomialCommitment`s."
    );

    // We reuse the LDE computed in `ListPolynomialCommitment` and extract every `step` points to get
    // an LDE matching `max_filtered_constraint_degree`.
    let step = 1 << (common_data.config.rate_bits - max_degree_bits);
    // When opening the `Z`s polys at the "next" point in Plonk, need to look at the point `next_step`
    // steps away since we work on an LDE of degree `max_filtered_constraint_degree`.
    let next_step = 1 << max_degree_bits;

    let points = F::two_adic_subgroup(common_data.degree_bits + max_degree_bits);
    let lde_size = points.len();

    // Retrieve the LDE values at index `i`.
    let get_at_index = |comm: &'a ListPolynomialCommitment<F>, i: usize| -> &'a [F] {
        comm.get_lde_values(i * step)
    };

    let z_h_on_coset = ZeroPolyOnCoset::new(common_data.degree_bits, max_degree_bits);

    let quotient_values: Vec<Vec<F>> = points
        .into_par_iter()
        .enumerate()
        .map(|(i, x)| {
            let shifted_x = F::coset_shift() * x;
            let i_next = (i + next_step) % lde_size;
            let local_constants_sigmas = get_at_index(&prover_data.constants_sigmas_commitment, i);
            let local_constants = &local_constants_sigmas[common_data.constants_range()];
            let s_sigmas = &local_constants_sigmas[common_data.sigmas_range()];
            let local_wires = get_at_index(wires_commitment, i);
            let local_zs_partial_products = get_at_index(zs_partial_products_commitment, i);
            let local_zs = &local_zs_partial_products[common_data.zs_range()];
            let next_zs =
                &get_at_index(zs_partial_products_commitment, i_next)[common_data.zs_range()];
            let partial_products = &local_zs_partial_products[common_data.partial_products_range()];

            debug_assert_eq!(local_wires.len(), common_data.config.num_wires);
            debug_assert_eq!(local_zs.len(), num_challenges);

            let vars = EvaluationVarsBase {
                local_constants,
                local_wires,
            };
            let mut quotient_values = eval_vanishing_poly_base(
                common_data,
                i,
                shifted_x,
                vars,
                local_zs,
                next_zs,
                partial_products,
                s_sigmas,
                betas,
                gammas,
                alphas,
                &z_h_on_coset,
            );
            let denominator_inv = z_h_on_coset.eval_inverse(i);
            quotient_values
                .iter_mut()
                .for_each(|v| *v *= denominator_inv);
            quotient_values
        })
        .collect();

    transpose(&quotient_values)
        .into_par_iter()
        .map(PolynomialValues::new)
        .map(|values| values.coset_ifft(F::coset_shift()))
        .collect()
}
