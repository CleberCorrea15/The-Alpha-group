"""
SPECTRAL AND RECURRENCE AUDIT — ALPHA GROUP

Operators: B5, B8, B13, B15, B8^2, and B8^4.

Self-contained script for local Python or Google Colab.
The explicit matrices are taken from the paper's appendix.

Checks:
    1. Characteristic and minimal polynomials;
    2. Relevant quadratic factors;
    3. Exact spectral projectors;
    4. Idempotency, commutation, and rank;
    5. Total and projected traces;
    6. Linear recurrences;
    7. Relations to Lucas numbers.

Dependencies:
    pip install sympy pandas

Run:
    python alpha_group_spectral_recurrence_audit.py
"""


# ================================================================
# SPECTRAL AND RECURRENCE AUDIT — ALPHA GROUP
# Operators: B5, B8, B13, B15, B8^2, and B8^4
#
# Self-contained script for local Python or Google Colab.
# Explicit matrices as given in the paper's appendix.
#
# Checks:
#   1. Characteristic and minimal polynomials;
#   2. Relevant quadratic factors;
#   3. Exact spectral projectors;
#   4. Idempotency, commutation, and rank;
#   5. Total and projected traces;
#   6. Linear recurrences;
#   7. Relations to Lucas numbers.
# ================================================================

import sympy as sp
import pandas as pd

# Compatível com execução local e Google Colab; não depende de IPython.

x = sp.symbols("x")

# ================================================================
# 1. ORIGINAL MATRICES FROM THE ALPHA BASIS
# ================================================================

B5 = sp.Matrix([
    [1,  0,  0, 1],
    [0,  1,  1, 0],
    [0, -1,  0, 0],
    [1,  0,  0, 0],
])

B8 = sp.Matrix([
    [ 0, -1, -1, 0],
    [ 1,  0,  0, 1],
    [-1,  0,  0, 0],
    [ 0, -1,  0, 0],
])

B13 = sp.Matrix([
    [2,  0,  0, 1],
    [0,  0,  1, 0],
    [0, -1, -1, 0],
    [1,  0,  0, 1],
])

B15 = sp.Matrix([
    [0, 0,  0, -1],
    [0, 0,  1,  0],
    [0, 1, -1,  0],
    [1, 0,  0, -1],
])

# Operatores derivados, sem redefinição de matrizes.
B8_2 = B8**2
B8_4 = B8**4

MATRIZES = {
    "B5": B5,
    "B8": B8,
    "B13": B13,
    "B15": B15,
    "B8^2": B8_2,
    "B8^4": B8_4,
}

# ================================================================
# 2. QUADRATIC FACTORS OF INTEREST
# ================================================================

QUAD_FACTORS = {
    "x^2 - x - 1":
        sp.Poly(x**2 - x - 1, x, domain=sp.QQ),

    "x^2 - 3x + 1":
        sp.Poly(x**2 - 3*x + 1, x, domain=sp.QQ),

    "x^2 + x - 1":
        sp.Poly(x**2 + x - 1, x, domain=sp.QQ),
}

ZERO_POLY = sp.Poly(0, x, domain=sp.QQ)

# ================================================================
# 3. EXACT MINIMAL POLYNOMIAL
# ================================================================

def minimal_poly(A, max_degree=16):
    """
    Determines the minimal polynomial by exact linear dependence
    among I, A, A^2, ..., A^d.
    """

    n_entries = A.rows * A.cols
    powers = [sp.eye(A.rows)]

    for degree in range(1, max_degree + 1):

        powers.append(sp.simplify(powers[-1] * A))

        cols = sp.Matrix.hstack(
            *[P.reshape(n_entries, 1) for P in powers]
        )

        if cols.rank() < len(powers):

            null_vectors = cols.nullspace()

            for vec in null_vectors:

                lead = sp.simplify(vec[-1])

                if lead != 0:

                    coeffs = [
                        sp.simplify(c / lead)
                        for c in vec
                    ]

                    expr = sum(
                        coeffs[k] * x**k
                        for k in range(len(coeffs))
                    )

                    return sp.Poly(
                        sp.factor(expr),
                        x,
                        domain=sp.QQ
                    )

    raise RuntimeError(
        "Minimal polynomial not found."
    )


# ================================================================
# 4. MATRIX POLYNOMIAL EVALUATION — HORNER'S METHOD
# ================================================================

def eval_poly_matrix(poly_expr, M):
    """
    Computes p(M) exactly using Horner's method.
    """

    p = sp.Poly(
        sp.expand(poly_expr),
        x,
        domain=sp.QQ
    )

    acc = sp.zeros(M.rows, M.cols)
    identity = sp.eye(M.rows)

    for coeff in p.all_coeffs():
        acc = sp.simplify(acc * M + coeff * identity)

    return sp.simplify(acc)


# ================================================================
# 5. SPECTRAL PROJECTOR
# ================================================================

def spectral_projector_for_factor(M, min_poly, q_poly):
    """
    Constructs the projector associated with factor q of:

        m(x) = q(x) r(x)

    when gcd(q,r)=1.

    By the Bezout identity:

        s(x)q(x) + t(x)r(x) = 1

    The projector onto the sector associated with q is:

        P_q = t(M) r(M)

    If m=q, the entire space belongs to the sector and P_q=I.
    """

    m_poly = sp.Poly(
        min_poly.as_expr(),
        x,
        domain=sp.QQ
    )

    q_p = sp.Poly(
        q_poly.as_expr(),
        x,
        domain=sp.QQ
    )

    quo_poly, rem_poly = sp.div(m_poly, q_p)

    if rem_poly != ZERO_POLY:
        return None

    # Caso em que o polinômio mínimo é exatamente q.
    if quo_poly.degree() == 0:
        return sp.eye(M.rows)

    r_p = sp.Poly(
        quo_poly.as_expr(),
        x,
        domain=sp.QQ
    )

    # SymPy retorna s, t, h:
    # s*q + t*r = h.
    s_poly, t_poly, h_poly = sp.gcdex(q_p, r_p)

    if h_poly.degree() != 0:
        return None

    # Normalize so that s*q + t*r = 1.
    h_value = h_poly.as_expr()

    s_poly = sp.Poly(
        s_poly.as_expr() / h_value,
        x,
        domain=sp.QQ
    )

    t_poly = sp.Poly(
        t_poly.as_expr() / h_value,
        x,
        domain=sp.QQ
    )

    # Projetor P_q = t(M) r(M).
    t_M = eval_poly_matrix(t_poly.as_expr(), M)
    r_M = eval_poly_matrix(r_p.as_expr(), M)

    P_q = sp.simplify(t_M * r_M)

    return P_q


# ================================================================
# 6. LUCAS SEQUENCE
# ================================================================

def lucas_seq(length=30):

    L = [2, 1]

    while len(L) < length:
        L.append(L[-1] + L[-2])

    return L


N_TERMS = 13

L_full = lucas_seq(2 * N_TERMS + 2)

L_even = [
    L_full[2*n]
    for n in range(N_TERMS)
]

L_signed = [
    (-1)**n * L_full[n]
    for n in range(N_TERMS)
]

L_signed_double = [
    2 * (-1)**n * L_full[n]
    for n in range(N_TERMS)
]

L_even_double = [
    2 * L_full[2*n]
    for n in range(N_TERMS)
]


# ================================================================
# 7. RECURRENCE VERIFICATION
# ================================================================

COEFFS_MAP = {

    "x^2 - x - 1": (1, 1),

    "x^2 - 3x + 1": (3, -1),

    "x^2 + x - 1": (-1, 1),
}


def check_recurrence(seq, q_name):

    c1, c0 = COEFFS_MAP[q_name]

    return all(
        sp.simplify(
            seq[n + 2] -
            (c1 * seq[n + 1] + c0 * seq[n])
        ) == 0

        for n in range(len(seq) - 2)
    )


# ================================================================
# 8. CHARACTERISTIC AND MINIMAL POLYNOMIALS
# ================================================================

poly_rows = []

MIN_POLYS = {}
CHAR_POLYS = {}

for name, M in MATRIZES.items():

    cp = sp.Poly(
        M.charpoly(x).as_expr(),
        x,
        domain=sp.QQ
    ).monic()

    mp = minimal_poly(M)

    CHAR_POLYS[name] = cp
    MIN_POLYS[name] = mp

    matched_factors = [
        q_label
        for q_label, q_poly in QUAD_FACTORS.items()
        if sp.rem(mp, q_poly) == ZERO_POLY
    ]

    poly_rows.append({

        "Matrix": name,

        "Characteristic polynomial":
            str(sp.factor(cp.as_expr())),

        "Minimal polynomial":
            str(sp.factor(mp.as_expr())),

        "Quadratic factors present":
            ", ".join(matched_factors)
            if matched_factors else "-"
    })


df_polys = pd.DataFrame(poly_rows)

print("\n" + "="*75)
print("CHARACTERISTIC AND MINIMAL POLYNOMIALS")
print("="*75)

print(df_polys.to_string(index=False))


# ================================================================
# 9. TRACES, PROJECTORS, AND RECURRENCES
# ================================================================

rec_rows = []
seq_details = {}
projector_checks = []

for name, M in MATRIZES.items():

    mp = MIN_POLYS[name]

    # Successive powers.
    powers_M = [sp.eye(M.rows)]

    for _ in range(1, N_TERMS):

        powers_M.append(
            sp.simplify(powers_M[-1] * M)
        )

    # Total trace.
    tr_total = [
        sp.simplify(P.trace())
        for P in powers_M
    ]

    for q_label, q_poly in QUAD_FACTORS.items():

        if sp.rem(mp, q_poly) != ZERO_POLY:
            continue

        P_q = spectral_projector_for_factor(
            M,
            mp,
            q_poly
        )

        if P_q is None:

            rec_rows.append({

                "Object and factor":
                    f"{name} | {q_label}",

                "Rank":
                    "N/A",

                "Valid projector":
                    False,

                "Sector recurrence":
                    "N/A",

                "Total-trace recurrence":
                    check_recurrence(tr_total, q_label),

                "Reference":
                    "Not calculated"
            })

            continue

        # Projected trace.
        tr_proj = [
            sp.simplify(
                (powers_M[n] * P_q).trace()
            )
            for n in range(N_TERMS)
        ]

        # Auditoria do projetor.
        idempotence_ok = (
            sp.simplify(P_q**2 - P_q) == sp.zeros(4)
        )

        commutation_ok = (
            sp.simplify(P_q*M - M*P_q) == sp.zeros(4)
        )

        projector_rank = P_q.rank()

        projector_checks.append({

            "Operator": name,

            "Factor": q_label,

            "Idempotency":
                idempotence_ok,

            "Commutes with M":
                commutation_ok,

            "Rank":
                projector_rank
        })

        rec_proj_ok = check_recurrence(
            tr_proj,
            q_label
        )

        rec_total_ok = check_recurrence(
            tr_total,
            q_label
        )

        # References específicas.
        if name == "B5" and q_label == "x^2 - x - 1":

            reference = L_full[:N_TERMS]
            ref_name = "Lucas L_n"

        elif name == "B13" and q_label == "x^2 - 3x + 1":

            reference = L_even
            ref_name = "Lucas L_2n"

        elif name == "B15" and q_label == "x^2 + x - 1":

            reference = L_signed
            ref_name = "(-1)^n L_n"

        elif name == "B8^2" and q_label == "x^2 + x - 1":

            reference = L_signed_double
            ref_name = "2(-1)^n L_n"

        elif name == "B8^4" and q_label == "x^2 - 3x + 1":

            reference = L_even_double
            ref_name = "2 L_2n"

        else:

            reference = None
            ref_name = "No specific comparison"

        if reference is not None:
            same_as_ref = (tr_proj == reference)
        else:
            same_as_ref = False

        key = f"{name} | {q_label}"

        seq_details[key] = pd.DataFrame({

            "n": list(range(N_TERMS)),

            "Total trace": tr_total,

            "Projected trace": tr_proj,

            "Lucas L_n": L_full[:N_TERMS],

            "Lucas L_2n": L_even,

            "(-1)^n L_n": L_signed
        })

        rec_rows.append({

            "Object and factor": key,

            "Rank": projector_rank,

            "Valid projector":
                idempotence_ok and commutation_ok,

            "Sector recurrence":
                rec_proj_ok,

            "Total-trace recurrence":
                rec_total_ok,

            "Reference":
                ref_name,

            "Matches reference":
                same_as_ref,

            "First terms":
                tr_proj[:8]
        })


# ================================================================
# 10. RECURRENCE TABLE
# ================================================================

df_rec = pd.DataFrame(rec_rows)

print("\n" + "="*75)
print("RECURRENCE SUMMARY BY QUADRATIC FACTOR")
print("="*75)

print(df_rec.to_string(index=False))


# ================================================================
# 11. PROJECTOR AUDIT
# ================================================================

df_projectors = pd.DataFrame(projector_checks)

print("\n" + "="*75)
print("SPECTRAL PROJECTOR AUDIT")
print("="*75)

print(df_projectors.to_string(index=False))


# ================================================================
# 12. FULL SEQUENCES
# ================================================================

for key, tbl in seq_details.items():

    print("\n" + "="*75)
    print("SEQUENCE:", key)
    print("="*75)

    print(tbl.to_string(index=False))


# ================================================================
# 13. TESTS OF RELATIONS BETWEEN SEQUENCES
# ================================================================

print("\n" + "="*75)
print("LUCAS RELATION TESTS")
print("="*75)

tests = {

    "B5 sector = L_n":
        L_full[:N_TERMS],

    "B13 sector = L_2n":
        L_even,

    "B15 sector = (-1)^n L_n":
        L_signed,

    "B8^2 total = 2(-1)^n L_n":
        L_signed_double,

    "B8^4 total = 2 L_2n":
        L_even_double,
}

# Retrieve the calculated sequences.
calculated = {}

for key, tbl in seq_details.items():

    calculated[key] = list(
        tbl["Projected trace"]
    )

# Specific comparisons.
comparison_pairs = [

    ("B5 | x^2 - x - 1", "B5 sector = L_n"),

    ("B13 | x^2 - 3x + 1", "B13 sector = L_2n"),

    ("B15 | x^2 + x - 1", "B15 sector = (-1)^n L_n"),

    ("B8^2 | x^2 + x - 1", "B8^2 total = 2(-1)^n L_n"),

    ("B8^4 | x^2 - 3x + 1", "B8^4 total = 2 L_2n"),
]

for calculated_key, reference_key in comparison_pairs:

    if calculated_key in calculated:

        result = (
            calculated[calculated_key]
            == tests[reference_key]
        )

        print(
            f"{calculated_key}: "
            f"{'PASSED' if result else 'FAILED'}"
        )


# ================================================================
# 14. EXACT IDENTITIES BETWEEN SEQUENCES
# ================================================================

print("\n" + "="*75)
print("IDENTITIES BETWEEN THE SEQUENCES")
print("="*75)

if "B13 | x^2 - 3x + 1" in calculated:

    b13_seq = calculated[
        "B13 | x^2 - 3x + 1"
    ]

    b8_4_seq = calculated[
        "B8^4 | x^2 - 3x + 1"
    ]

    print(
        "B8^4: tr(B8^(4n)) = 2 * B13 sector:",
        b8_4_seq == [2*v for v in b13_seq]
    )

if "B15 | x^2 + x - 1" in calculated:

    b15_seq = calculated[
        "B15 | x^2 + x - 1"
    ]

    b8_2_seq = calculated[
        "B8^2 | x^2 + x - 1"
    ]

    print(
        "B8^2: tr(B8^(2n)) = 2 * B15 sector:",
        b8_2_seq == [2*v for v in b15_seq]
    )


# ================================================================
# 15. FINAL SUMMARY
# ================================================================

print("\n" + "="*75)
print("AUDIT COMPLETED")
print("="*75)

print("Number of operators:", len(MATRIZES))

print(
    "Projectors audited:",
    len(df_projectors)
)

print(
    "All projectors are idempotent:",
    all(df_projectors["Idempotency"])
)

print(
    "All projectors commute with their operators:",
    all(df_projectors["Commutes with M"])
)

print(
    "All reference tests passed:",
    all(
        row["Matches reference"]
        for row in rec_rows
        if row.get("Reference") not in (
            None,
            "No specific comparison",
            "Not calculated"
        )
    )
)

print("\nEnd of spectral audit.")
