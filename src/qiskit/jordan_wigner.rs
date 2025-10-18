//! Jordan-Wigner transformation.
//!
//! Transform a fermion operator to a qubit operator using the Jordan-Wigner
//! transformation. The Jordan-Wigner transformation maps fermionic annihilation
//! operators to qubits as follows:
//!
//! .. math::
//!
//!     a_p \mapsto \frac12 (X_p + iY_p)Z_1 \cdots Z_{p-1}
//!
//! In the transformed operator, the first ``norb`` qubits represent spin-up (alpha)
//! orbitals, and the latter ``norb`` qubits represent spin-down (beta) orbitals. As a
//! result of this convention, the qubit index that an orbital is mapped to depends on
//! the total number of spatial orbitals. By default, the total number of spatial
//! orbitals is automatically determined by the largest-index orbital present in the
//! operator, but you can manually specify the number using the `norb` argument.

use std::collections::HashMap;

use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActionTriplet {
    /// true => creation, false => annihilation
    action: bool,
    /// spin index 0 (alpha) or 1 (beta)
    spin: usize,
    /// spatial orbital index
    orb: usize,
}

#[derive(Debug, Clone)]
struct JWTerm {
    ops: Vec<ActionTriplet>,
    coeff: Complex64,
}

#[derive(Debug, Clone)]
struct TwoBranchOp {
    x_str: String,
    x_coeff: Complex64, // +- 1/2
    y_str: String,
    y_coeff: Complex64, // +- i/2
}

fn jordan_wigner_terms(
    fermion_terms: &[JWTerm],
    norb_opt: Option<usize>,
) -> Result<(usize, Vec<(String, Complex64)>), String> {
    if fermion_terms.is_empty() {
        let n = 2 * norb_opt.unwrap_or(0);
        return Ok((n, Vec::new()));
    }
    let norb_in_op = 1 + fermion_terms
        .iter()
        .flat_map(|t| t.ops.iter().map(|x| x.orb))
        .max()
        .unwrap_or(0);
    let norb = norb_opt.unwrap_or(norb_in_op);

    if norb < norb_in_op {
        return Err(format!(
            "Number of spatial orbitals specified is fewer than detected in the operator. \
             Operator needs at least {norb_in_op}, got {norb}."
        ));
    }
    let num_qubits = 2 * norb;

    // Cache per-qubit primitive
    let mut cache: HashMap<(bool, usize, usize), TwoBranchOp> = HashMap::new();

    // Expand in stable order
    let mut expanded: Vec<(String, Complex64)> = Vec::new();

    for term in fermion_terms {
        // Start from identity string with term coeff
        let mut acc: Vec<(String, Complex64)> = vec![("I".repeat(num_qubits), term.coeff)];

        for &ActionTriplet { action, spin, orb } in &term.ops {
            let qubit = orb + spin * norb;
            let prim = get_two_branch(action, qubit, norb, num_qubits, &mut cache);

            // acc @ prim: append X-branch first, then Y-branch to match Python ref
            let mut next: Vec<(String, Complex64)> = Vec::with_capacity(acc.len() * 2);
            for (l_str, l_c) in acc.into_iter() {
                // X branch
                next.push(pauli_mul(&l_str, l_c, &prim.x_str, prim.x_coeff));
                // Y branch
                next.push(pauli_mul(&l_str, l_c, &prim.y_str, prim.y_coeff));
            }
            acc = next;
        }
        expanded.extend(acc.into_iter());
    }

    // Stable simplify: merge identical strings keeping first-seen order
    let mut idx: HashMap<String, usize> = HashMap::new();
    let mut unique: Vec<(String, Complex64)> = Vec::new();
    for (s, c) in expanded.into_iter() {
        if c.norm() < EPS {
            continue;
        }
        if let Some(&i) = idx.get(&s) {
            unique[i].1 += c;
        } else {
            let i = unique.len();
            idx.insert(s.clone(), i);
            unique.push((s, c));
        }
    }
    unique.retain(|(_, c)| c.norm() >= EPS);

    Ok((num_qubits, unique))
}

fn get_two_branch(
    action: bool,
    qubit: usize,
    _norb: usize,
    num_qubits: usize,
    cache: &mut HashMap<(bool, usize, usize), TwoBranchOp>,
) -> TwoBranchOp {
    if let Some(op) = cache.get(&(action, qubit, num_qubits)) {
        return op.clone();
    }
    let mut xs = vec!['I'; num_qubits];
    let mut ys = vec!['I'; num_qubits];
    for i in 0..qubit {
        xs[i] = 'Z';
        ys[i] = 'Z';
    }
    xs[qubit] = 'X';
    ys[qubit] = 'Y';

    let x_coeff = Complex64::new(0.5, 0.0);
    // Python convention: creation => -0.5j ; annihilation => +0.5j
    let y_coeff = if action {
        Complex64::new(0.0, -0.5)
    } else {
        Complex64::new(0.0, 0.5)
    };

    let op = TwoBranchOp {
        x_str: xs.into_iter().collect(),
        x_coeff,
        y_str: ys.into_iter().collect(),
        y_coeff,
    };
    cache.insert((action, qubit, num_qubits), op.clone());
    op
}

#[inline]
fn pauli_mul(left: &str, l_c: Complex64, right: &str, r_c: Complex64) -> (String, Complex64) {
    let mut out = String::with_capacity(left.len());
    let mut phase = Complex64::new(1.0, 0.0);
    for (a, b) in left.bytes().zip(right.bytes()) {
        let (p, ph) = mul1(a as char, b as char);
        out.push(p);
        phase *= ph;
    }
    (out, l_c * r_c * phase)
}

#[inline]
fn mul1(a: char, b: char) -> (char, Complex64) {
    match (a, b) {
        ('I', 'I') => ('I', Complex64::new(1.0, 0.0)),
        ('I', 'X') => ('X', Complex64::new(1.0, 0.0)),
        ('I', 'Y') => ('Y', Complex64::new(1.0, 0.0)),
        ('I', 'Z') => ('Z', Complex64::new(1.0, 0.0)),
        ('X', 'I') => ('X', Complex64::new(1.0, 0.0)),
        ('Y', 'I') => ('Y', Complex64::new(1.0, 0.0)),
        ('Z', 'I') => ('Z', Complex64::new(1.0, 0.0)),

        ('X', 'X') => ('I', Complex64::new(1.0, 0.0)),
        ('Y', 'Y') => ('I', Complex64::new(1.0, 0.0)),
        ('Z', 'Z') => ('I', Complex64::new(1.0, 0.0)),

        // Non-commuting pairs with ±i phases (right multiplication)
        ('X', 'Y') => ('Z', Complex64::new(0.0, 1.0)), // i Z
        ('Y', 'X') => ('Z', Complex64::new(0.0, -1.0)), // -i Z
        ('Y', 'Z') => ('X', Complex64::new(0.0, 1.0)), // i X
        ('Z', 'Y') => ('X', Complex64::new(0.0, -1.0)), // -i X
        ('Z', 'X') => ('Y', Complex64::new(0.0, 1.0)), // i Y
        ('X', 'Z') => ('Y', Complex64::new(0.0, -1.0)), // -i Y
        (u, v) => panic!("Invalid Pauli pair: {u}{v}"),
    }
}

// --------- PyO3 export: qiskit.jordan_wigner.jordan_wigner ---------
#[pyfunction]
pub fn jw_map(py: Python<'_>, op: &Bound<PyAny>, norb: Option<usize>) -> PyResult<PyObject> {
    // 1) Iterate op.items() inside Rust to avoid Python-level loops.
    let items_iter = op.call_method0("items")?;
    let iter = items_iter.iter()?;

    // Collect into plain Rust structures so we can release the GIL during heavy work
    let mut terms: Vec<JWTerm> = Vec::new();
    for obj in iter {
        // obj: PyResult<Bound<PyAny>>
        let any: Bound<PyAny> = obj?;
        let pair_ref: &Bound<PyTuple> = any.downcast()?; // downcast returns &Bound<PyTuple>

        if pair_ref.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "FermionOperator.items() must yield (key, coeff) pairs",
            ));
        }
        let key_any = pair_ref.get_item(0)?;
        let coeff_any = pair_ref.get_item(1)?;
        let coeff: Complex64 = coeff_any.extract()?;

        let key_tup_ref: &Bound<PyTuple> = key_any.downcast()?;
        let mut ops: Vec<ActionTriplet> = Vec::with_capacity(key_tup_ref.len());

        // PyTuple::iter() yields &Bound<PyAny>; clone before downcasting
        for elt in key_tup_ref.iter() {
            let elt_any: Bound<PyAny> = elt.clone();
            let t_ref: &Bound<PyTuple> = elt_any.downcast()?;
            if t_ref.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Each operator in the key must be a 3-tuple: (action: bool, spin_bool: bool, orb: int)",
                ));
            }
            let action: bool = t_ref.get_item(0)?.extract()?;
            let spin_bool: bool = t_ref.get_item(1)?.extract()?;
            let orb: usize = t_ref.get_item(2)?.extract()?;
            let spin = if spin_bool { 1 } else { 0 };
            ops.push(ActionTriplet { action, spin, orb });
        }
        terms.push(JWTerm { ops, coeff });
    }

    // 2) Release the GIL for expansion+simplify (the heavy part)
    let (num_qubits, merged_terms) = py
        .allow_threads(|| jordan_wigner_terms(&terms, norb))
        .map_err(pyo3::exceptions::PyValueError::new_err)?; // <- clippy fix

    // 3) Build qiskit.quantum_info.SparsePauliOp.from_sparse_list, then simplify.
    let qi = py.import_bound("qiskit.quantum_info")?;
    let spo_cls = qi.getattr("SparsePauliOp")?;

    let qubits_all: Vec<usize> = (0..num_qubits).collect();
    let triplets = PyList::empty_bound(py);
    for (s, c) in merged_terms.iter() {
        // Each entry is (pauli_string, qubit_indices, coeff)
        let item = (s.as_str(), &qubits_all, *c);
        triplets.append(item)?;
    }

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("num_qubits", num_qubits)?;
    let py_spo = spo_cls.call_method("from_sparse_list", (triplets,), Some(&kwargs))?;
    let py_spo = py_spo.call_method0("simplify")?;

    Ok(py_spo.into_py(py))
}
