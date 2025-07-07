#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};
#[cfg(not(feature = "cuda"))]
use std::alloc;
#[cfg(feature = "cuda")]
use std::alloc::{AllocError, Allocator, Layout};
#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use plonky2_cuda;
#[cfg(feature = "cuda")]
use rustacuda::prelude::*;
#[cfg(feature = "cuda")]
use rustacuda::memory::{AsyncCopyDestination, DeviceBuffer, DeviceSlice, cuda_malloc_locked, cuda_free_locked};
#[cfg(feature = "cuda")]
use std::mem::transmute;
#[cfg(feature = "cuda")]
use std::mem;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::ptr::NonNull;
#[cfg(feature = "cuda")]
use crate::plonk::config::Hasher;
#[cfg(feature = "cuda")]
use crate::hash::merkle_tree::MerkleCap;

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CUDAAllocator {}

#[cfg(feature = "cuda")]
unsafe impl Allocator for CUDAAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let raw_ptr = cuda_malloc_locked::<u8>(layout.size()).unwrap();
            let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
            Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: `layout` is non-zero in size,
            // other conditions must be upheld by the caller
            unsafe {
                // dealloc(ptr.as_ptr(), layout)
                cuda_free_locked(ptr.as_ptr()).unwrap();
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaInnerContext {
    pub stream: rustacuda::stream::Stream,
    pub stream2: rustacuda::stream::Stream,

}

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Debug)]
pub struct CudaInvContext<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub inner: CudaInnerContext,
    pub ext_values_flatten :Arc<Vec<F>>,
    pub values_flatten     :Arc<Vec<F, CUDAAllocator>>,
    pub digests_and_caps_buf :Arc<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>,

    pub ext_values_flatten2 :Arc<Vec<F>>,
    pub values_flatten2     :Arc<Vec<F, CUDAAllocator>>,
    pub digests_and_caps_buf2 :Arc<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>,

    pub ext_values_flatten3 :Arc<Vec<F>>,
    pub values_flatten3     :Arc<Vec<F, CUDAAllocator>>,
    pub digests_and_caps_buf3 :Arc<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>,

    // pub values_device: DeviceBuffer::<F>,
    // pub ext_values_device: DeviceBuffer::<F>,
    pub cache_mem_device: DeviceBuffer::<F>,
    pub second_stage_offset: usize,

    pub root_table_device: DeviceBuffer::<F>,
    pub root_table_device2: DeviceBuffer::<F>,
    pub constants_sigmas_commitment_leaves_device: DeviceBuffer::<F>,
    pub shift_powers_device: DeviceBuffer::<F>,
    pub shift_inv_powers_device: DeviceBuffer::<F>,

    pub points_device: DeviceBuffer::<F>,
    pub z_h_on_coset_evals_device: DeviceBuffer::<F>,
    pub z_h_on_coset_inverses_device: DeviceBuffer::<F>,
    pub k_is_device: DeviceBuffer::<F>,

    pub ctx: Context,
}

#[cfg(not(feature = "cuda"))]
#[repr(C)]
#[derive(Debug)]
pub struct CudaInvContext<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> {
    pub _p : std::marker::PhantomData<F>,
    pub _c : std::marker::PhantomData<C>,
}

#[cfg(not(feature = "cuda"))]
/// Alias Global allocator to CUDAAllocator
pub type CUDAAllocator = alloc::Global;

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();
        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.leaves[index];
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        #[cfg(feature = "cuda")]
        ctx: &mut Option<&mut crate::fri::oracle::CudaInvContext<F, C, D>>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
        );

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            #[cfg(feature = "cuda")]
            ctx,
            timing,
        );

        fri_proof
    }

    #[cfg(feature = "cuda")]
    pub fn from_values_with_gpu(
        values: &Vec<F>,
        num_polynomials: usize,
        degree: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
        fft_root_table_deg: &Vec<F>,
        ctx: &mut CudaInvContext<F, C, D>,
        stage: usize,
    ) -> Self {
        assert!(stage == 1 || stage == 2, "stage must be 1 or 2");

        let salt_size = if blinding { SALT_SIZE } else { 0 };

        let degree_log = log2_strict(degree);
        let n_inv = F::inverse_2exp(degree_log);
        let n_inv_ptr: *const F = &n_inv;

        let len_cap = (1 << cap_height);
        let num_digests = 2 * (degree * (1 << rate_bits) - len_cap);
        let num_digests_and_caps = num_digests + len_cap;

        let values_flatten_len = num_polynomials * degree;
        let ext_values_flatten_len = (values_flatten_len + salt_size * degree) * (1 << rate_bits);

        let pad_extvalues_len = ext_values_flatten_len;

        let (ext_values_flatten, values_flatten, digests_and_caps_buf);

        let ext_values_device_offset;
        if stage == 1 {
            ext_values_flatten = Arc::<Vec<F>>::get_mut(&mut ctx.ext_values_flatten).unwrap();
            values_flatten =
                Arc::<Vec<F, CUDAAllocator>>::get_mut(&mut ctx.values_flatten).unwrap();
            digests_and_caps_buf =
                Arc::<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>::get_mut(
                    &mut ctx.digests_and_caps_buf,
                )
                .unwrap();
            ext_values_device_offset = 0;
        } else {
            ext_values_flatten = Arc::<Vec<F>>::get_mut(&mut ctx.ext_values_flatten2).unwrap();
            values_flatten =
                Arc::<Vec<F, CUDAAllocator>>::get_mut(&mut ctx.values_flatten2).unwrap();
            digests_and_caps_buf =
                Arc::<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>::get_mut(
                    &mut ctx.digests_and_caps_buf2,
                )
                .unwrap();
            ext_values_device_offset = ctx.second_stage_offset;
        }

        let values_device = ctx
            .cache_mem_device
            .split_at_mut(ext_values_device_offset)
            .1;


        let root_table_device = &ctx.root_table_device;
        let root_table_device2 = &ctx.root_table_device2;
        let shift_powers_device = &ctx.shift_powers_device;

        timed!(timing, "copy values to gpu", unsafe {
            transmute::<&mut DeviceSlice<F>, &mut DeviceSlice<u64>>(
                &mut values_device[0..values_flatten_len],
            )
            .async_copy_from(transmute::<&Vec<F>, &Vec<u64>>(values), &ctx.inner.stream)
            .unwrap();
            ctx.inner.stream.synchronize().unwrap();
        });

        unsafe {
            let ctx_ptr: *mut CudaInnerContext = &mut ctx.inner;
            timed!(timing, "IFTT", {
                plonky2_cuda::ifft(
                    values_device.as_mut_ptr() as *mut u64,
                    num_polynomials as i32,
                    degree as i32,
                    degree_log as i32,
                    root_table_device.as_ptr() as *const u64,
                    n_inv_ptr as *const u64,
                    ctx_ptr as *mut core::ffi::c_void,
                );
            });
            timed!(timing, "FFT + build Merkle tree + transpose with gpu", {
                unsafe {
                    transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(
                        &values_device[0..values_flatten_len],
                    )
                    .async_copy_to(
                        transmute::<&mut Vec<F, CUDAAllocator>, &mut Vec<u64>>(values_flatten),
                        &ctx.inner.stream2,
                    )
                    .unwrap();
                }

                plonky2_cuda::merkle_tree_from_coeffs(
                    values_device.as_mut_ptr() as *mut u64,
                    values_device.as_mut_ptr() as *mut u64,
                    num_polynomials as i32,
                    degree as i32,
                    degree_log as i32,
                    root_table_device.as_ptr() as *const u64,
                    root_table_device2.as_ptr() as *const u64,
                    shift_powers_device.as_ptr() as *const u64,
                    rate_bits as i32,
                    salt_size as i32,
                    cap_height as i32,
                    pad_extvalues_len as i32,
                    ctx_ptr as *mut core::ffi::c_void,
                );
            });
        }

        timed!(timing, "copy result back to cpu", {
            let mut alllen = ext_values_flatten_len;
            assert!(ext_values_flatten.len() == ext_values_flatten_len);
            alllen += pad_extvalues_len;

            let len_with_F = num_digests_and_caps * 4;
            let fs = unsafe { mem::transmute::<&mut Vec<_>, &mut Vec<F>>(digests_and_caps_buf) };
            unsafe {
                fs.set_len(len_with_F);
            }
            unsafe {
                transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(
                    &values_device[alllen..alllen + len_with_F],
                )
                .async_copy_to(
                    transmute::<&mut Vec<F>, &mut Vec<u64>>(fs),
                    &ctx.inner.stream,
                )
                .unwrap();
                ctx.inner.stream.synchronize().unwrap();
            }
            unsafe {
                fs.set_len(len_with_F / 4);
            }
        });

        let coeffs = values_flatten
            .par_chunks(degree)
            .map(|chunk| PolynomialCoeffs {
                coeffs: chunk.to_vec(),
            })
            .collect::<Vec<_>>();

        {
            let polynomials = coeffs;
            let (ctx_ext_values_flatten, ctx_digests_and_caps_buf);
            if stage == 1 {
                ctx_ext_values_flatten = ctx.ext_values_flatten.clone();
                ctx_digests_and_caps_buf = ctx.digests_and_caps_buf.clone();
            } else {
                ctx_ext_values_flatten = ctx.ext_values_flatten2.clone();
                ctx_digests_and_caps_buf = ctx.digests_and_caps_buf2.clone();
            }

            let ctx_ext_values_flatten_len = ctx_ext_values_flatten.len();
            let merkle_tree = MerkleTree {
                leaves: vec![],
                digests: vec![],
                cap: MerkleCap(
                    ctx_digests_and_caps_buf[num_digests..num_digests_and_caps].to_vec(),
                ),
                leaf_len: num_polynomials + salt_size,
                flatten_leaves: ctx_ext_values_flatten,
                leaves_len: ctx_ext_values_flatten_len,
                device_offset: ext_values_device_offset as isize,
                digests_and_cap: ctx_digests_and_caps_buf,
            };

            Self {
                polynomials,
                merkle_tree,
                degree_log,
                rate_bits,
                blinding,
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_coeffs_with_gpu(
        degree: usize,
        num_polynomials: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        ctx: &mut CudaInvContext<F, C, D>,
        stage: usize,
        offset: usize,
    ) -> Self {
        assert!(stage == 3, "stage must be 3");

        let salt_size = if blinding { SALT_SIZE } else { 0 };

        let degree_log = log2_strict(degree);
        let n_inv = F::inverse_2exp(degree_log);
        let n_inv_ptr: *const F = &n_inv;

        let len_cap = (1 << cap_height);
        let num_digests = 2 * (degree * (1 << rate_bits) - len_cap);
        let num_digests_and_caps = num_digests + len_cap;

        let values_flatten_len = num_polynomials * degree;
        let ext_values_flatten_len = (values_flatten_len + salt_size * degree) * (1 << rate_bits);
        let digests_and_caps_buf_len = num_digests_and_caps;

        let pad_extvalues_len = ext_values_flatten_len;

        let values_flatten = Arc::<Vec<F, MyAllocator>>::get_mut(&mut ctx.values_flatten3).unwrap();
        let ext_values_flatten = Arc::<Vec<F>>::get_mut(&mut ctx.ext_values_flatten3).unwrap();
        let digests_and_caps_buf =
            Arc::<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>::get_mut(
                &mut ctx.digests_and_caps_buf3,
            )
            .unwrap();

        let ext_values_device_offset = ctx.second_stage_offset + offset;

        let values_device = ctx
            .cache_mem_device
            .split_at_mut(ext_values_device_offset)
            .1;

        let root_table_device = &ctx.root_table_device;
        let root_table_device2 = &ctx.root_table_device2;
        let shift_powers_device = &ctx.shift_powers_device;

        unsafe {
            let ctx_ptr: *mut CudaInnerContext = &mut ctx.inner;
            timed!(timing, "FFT + build Merkle tree + transpose with gpu", {
                unsafe {
                    transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(
                        &values_device[0..values_flatten_len],
                    )
                    .async_copy_to(
                        transmute::<&mut Vec<F, CUDAAllocator>, &mut Vec<u64>>(values_flatten),
                        &ctx.inner.stream2,
                    )
                    .unwrap();
                }

                plonky2_cuda::merkle_tree_from_coeffs(
                    values_device.as_mut_ptr() as *mut u64,
                    values_device.as_mut_ptr() as *mut u64,
                    num_polynomials as i32,
                    degree as i32,
                    degree_log as i32,
                    root_table_device.as_ptr() as *const u64,
                    root_table_device2.as_ptr() as *const u64,
                    shift_powers_device.as_ptr() as *const u64,
                    rate_bits as i32,
                    salt_size as i32,
                    cap_height as i32,
                    pad_extvalues_len as i32,
                    ctx_ptr as *mut core::ffi::c_void,
                );
            });
        }
        timed!(timing, "copy result back to cpu", {
            let mut alllen = ext_values_flatten_len;
            assert!(ext_values_flatten.len() == ext_values_flatten_len);
            alllen += pad_extvalues_len;

            let len_with_F = digests_and_caps_buf_len * 4;
            let fs = unsafe { mem::transmute::<&mut Vec<_>, &mut Vec<F>>(digests_and_caps_buf) };
            unsafe {
                fs.set_len(len_with_F);
            }
            unsafe {
                transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(
                    &values_device[alllen..alllen + len_with_F],
                )
                .async_copy_to(
                    transmute::<&mut Vec<F>, &mut Vec<u64>>(fs),
                    &ctx.inner.stream,
                )
                .unwrap();
                ctx.inner.stream.synchronize().unwrap();
            }
            unsafe {
                fs.set_len(len_with_F / 4);
            }
        });

        let coeffs = values_flatten
            .par_chunks(degree)
            .map(|chunk| PolynomialCoeffs {
                coeffs: chunk.to_vec(),
            })
            .collect::<Vec<_>>();

        {
            let polynomials = coeffs;
            let ctx_ext_values_flatten = ctx.ext_values_flatten.clone();
            let ctx_digests_and_caps_buf = ctx.digests_and_caps_buf3.clone();

            let ctx_ext_values_flatten_len = ext_values_flatten_len;
            let merkle_tree = MerkleTree {
                leaves: vec![],
                digests: vec![],
                cap: MerkleCap(
                    ctx_digests_and_caps_buf[num_digests..num_digests_and_caps].to_vec(),
                ),
                leaf_len: num_polynomials + salt_size,
                flatten_leaves: ctx_ext_values_flatten,
                leaves_len: ctx_ext_values_flatten_len,
                device_offset: ext_values_device_offset as isize,
                digests_and_cap: ctx_digests_and_caps_buf,
            };

            Self {
                polynomials,
                merkle_tree,
                degree_log,
                rate_bits,
                blinding,
            }
        }
    }
}
