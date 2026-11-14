import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Max
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Push

/-!
Adap-CTA: formal companion to Appendix.tex, 2026-11-09.
Toolchain: Lean 4.19.0; mathlib v4.19.0.
Check from a mathlib project with the imported modules built:
  lake env lean AdapCTA_Formal.lean

The code proves finite-game results, projected update identities, coverage
identities, coefficient bounds, Lyapunov closure, and finite-round telescoping.
The convergence section formalizes the implication from the analytic one-round
estimates derived in Appendix D to both rates; those estimates are explicit
hypotheses of the corresponding declarations.
The endpoint derivative argument and the exact threshold B(M,delta) from
Appendix B are proved in `manuscript_threshold`; strict dominance then gives
the unique Nash equilibrium. The multi-stage declaration checks the additive
tail-cost inequality under the action-independent continuation model.

Normalized variables in Appendix D:
  t = beta * eta, s = rho_g^(-2), A = sigma^2/(K*N), V = sigma^2/K;
  e = beta^2 * local drift, U = beta^2 * model-distance energy;
  F = beta * convex objective gap, f = beta * nonconvex objective value.
The letter A in the convergence section denotes oracle variance, independently
of the communication coefficient A used in the stage-game section.

Every theorem is audited with `#print axioms` at the end of the file.
-/

noncomputable section
namespace AdapCTAFormal
open scoped BigOperators
open Set



def q (A c x : ℝ) := Real.sqrt (x+c) * (Real.exp (-x)-A)
def h (A c x : ℝ) := Real.exp (-x)*(1-2*(x+c))-A

theorem h_deriv (A c x : ℝ) :
    HasDerivAt (h A c) (Real.exp (-x)*(2*(x+c)-3)) x := by
  have he := ((hasDerivAt_id x).neg).exp
  have hl : HasDerivAt (fun z : ℝ => 1-2*(z+c)) (-2) x := by
    convert (hasDerivAt_const x (1:ℝ)).sub (((hasDerivAt_id x).add_const c).const_mul 2) using 1 <;> ring
  convert (he.mul hl).sub_const A using 1 <;> dsimp [h] <;> ring

theorem q_deriv (A c x : ℝ) (hx : 0 < x+c) :
    HasDerivAt (q A c) (h A c x / (2*Real.sqrt (x+c))) x := by
  have hs := ((hasDerivAt_id x).add_const c).sqrt (ne_of_gt hx)
  have he := (((hasDerivAt_id x).neg).exp).sub_const A
  convert hs.mul he using 1
  have hr := Real.sq_sqrt (le_of_lt hx)
  have hr0 : Real.sqrt (x+c) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hx)
  dsimp [h]
  field_simp
  nlinarith [congrArg (fun z : ℝ => Real.exp (-x)*z) hr]

theorem h_nonpos_high (A c x : ℝ) (hA : 0 ≤ A) (hx : 1/2 ≤ x+c) :
    h A c x ≤ 0 := by
  have hm := mul_nonpos_of_nonneg_of_nonpos (Real.exp_nonneg (-x))
    (show 1-2*(x+c) ≤ 0 by linarith)
  dsimp [h]
  linarith

theorem h_strictAnti_low (A c : ℝ) :
    StrictAntiOn (h A c) (Iio (1/2-c)) := by
  apply strictAntiOn_of_deriv_neg (convex_Iio _)
  · intro x _
    exact (h_deriv A c x).continuousAt.continuousWithinAt
  · intro x hx
    simp only [interior_Iio, mem_Iio] at hx
    rw [(h_deriv A c x).deriv]
    exact mul_neg_of_pos_of_neg (Real.exp_pos _) (by linarith)

/-- The endpoint argument used in Appendix B; no interior minimum is assumed. -/
theorem q_endpoint_min (A c a b x : ℝ) (hA : 0 ≤ A)
    (ha : 0 < a+c) (hx : x ∈ Icc a b) :
    min (q A c a) (q A c b) ≤ q A c x := by
  have hanti := h_strictAnti_low A c
  have hpos : ∀ y, a ≤ y → 0 < y+c := by intro y hy; linarith
  by_cases hh : 0 < h A c x
  · have hxl : x < 1/2-c := by
      by_contra hn
      have := h_nonpos_high A c x hA (by linarith)
      linarith
    have hmono : MonotoneOn (q A c) (Icc a x) := by
      apply monotoneOn_of_deriv_nonneg (convex_Icc _ _)
      · intro y hy
        exact (q_deriv A c y (hpos y hy.1)).continuousAt.continuousWithinAt
      · intro y hy
        simp only [interior_Icc, mem_Ioo] at hy
        exact (q_deriv A c y (hpos y (le_of_lt hy.1))).differentiableAt.differentiableWithinAt
      · intro y hy
        simp only [interior_Icc, mem_Ioo] at hy
        rw [(q_deriv A c y (hpos y (le_of_lt hy.1))).deriv]
        have hh' := hanti.antitoneOn (show y ∈ Iio (1/2-c) by change y < _; linarith)
          hxl (le_of_lt hy.2)
        exact div_nonneg (by linarith) (by positivity)
    exact (min_le_left _ _).trans (hmono ⟨le_rfl, hx.1⟩ ⟨hx.1, le_rfl⟩ hx.1)
  · have hh0 : h A c x ≤ 0 := le_of_not_gt hh
    have hmono : AntitoneOn (q A c) (Icc x b) := by
      apply antitoneOn_of_deriv_nonpos (convex_Icc _ _)
      · intro y hy
        exact (q_deriv A c y (hpos y (hx.1.trans hy.1))).continuousAt.continuousWithinAt
      · intro y hy
        simp only [interior_Icc, mem_Ioo] at hy
        exact (q_deriv A c y (hpos y (hx.1.trans (le_of_lt hy.1)))).differentiableAt.differentiableWithinAt
      · intro y hy
        simp only [interior_Icc, mem_Ioo] at hy
        rw [(q_deriv A c y (hpos y (hx.1.trans (le_of_lt hy.1)))).deriv]
        have hyh : h A c y ≤ 0 := by
          by_cases hyl : y < 1/2-c
          · exact (hanti.antitoneOn (show x ∈ Iio (1/2-c) by change x < _; linarith)
              hyl (le_of_lt hy.1)).trans hh0
          · exact h_nonpos_high A c y hA (by linarith)
        exact div_nonpos_of_nonpos_of_nonneg hyh (by positivity)
    exact (min_le_right _ _).trans (hmono ⟨le_rfl, hx.2⟩ ⟨hx.2, le_rfl⟩ hx.2)

def sectionCost (A M s x : ℝ) :=
  Real.exp (-x)+A*(1+x)+Real.sqrt (x+s)/Real.sqrt M

theorem sectionCost_deriv (A M s x : ℝ) (hx : 0 < x+s) :
    HasDerivAt (sectionCost A M s)
      (-Real.exp (-x)+A+1/(2*Real.sqrt (x+s))/Real.sqrt M) x := by
  have he := ((hasDerivAt_id x).neg).exp
  have hl := ((hasDerivAt_id x).const_add 1).const_mul A
  have hs := (((hasDerivAt_id x).add_const s).sqrt (ne_of_gt hx)).div_const (Real.sqrt M)
  convert (he.add hl).add hs using 1 <;> dsimp [sectionCost] <;> ring

/-- Endpoint sufficient condition, with c the smallest possible opponents' sum. -/
theorem endpoint_threshold_strict_decrease (A M c s a b : ℝ)
    (hA : 0 ≤ A) (hM : 0 < M) (ha : 0 < a+c) (hab : a ≤ b) (hcs : c ≤ s)
    (hleft : A < Real.exp (-a)-1/(2*Real.sqrt M*Real.sqrt (a+c)))
    (hright : A < Real.exp (-b)-1/(2*Real.sqrt M*Real.sqrt (b+c))) :
    StrictAntiOn (sectionCost A M s) (Icc a b) := by
  have hrootM : 0 < Real.sqrt M := Real.sqrt_pos.2 hM
  have hpos : ∀ x, a ≤ x → 0 < x+c := by intro x hx; linarith
  have hendpoint : ∀ x, a ≤ x →
      A < Real.exp (-x)-1/(2*Real.sqrt M*Real.sqrt (x+c)) →
      1/(2*Real.sqrt M) < q A c x := by
    intro x hx hbound
    have hr : 0 < Real.sqrt (x+c) := Real.sqrt_pos.2 (hpos x hx)
    have hb : (1/(2*Real.sqrt M))/Real.sqrt (x+c) < Real.exp (-x)-A := by
      have heq : (1/(2*Real.sqrt M))/Real.sqrt (x+c) =
          1/(2*Real.sqrt M*Real.sqrt (x+c)) := by ring
      rw [heq]
      linarith
    have hm := (div_lt_iff₀ hr).mp hb
    simpa only [q, mul_comm] using hm
  have hl := hendpoint a le_rfl hleft
  have hr := hendpoint b hab hright
  apply strictAntiOn_of_deriv_neg (convex_Icc _ _)
  · intro x hx
    exact (sectionCost_deriv A M s x (by have := hpos x hx.1; linarith)).continuousAt.continuousWithinAt
  · intro x hx
    simp only [interior_Icc, mem_Ioo] at hx
    have hxc := hpos x (le_of_lt hx.1)
    have hxs : 0 < x+s := by linarith
    rw [(sectionCost_deriv A M s x hxs).deriv]
    have hq : 1/(2*Real.sqrt M) < q A c x :=
      (lt_min hl hr).trans_le (q_endpoint_min A c a b x hA ha ⟨le_of_lt hx.1, le_of_lt hx.2⟩)
    have hr0 : 0 < Real.sqrt (x+c) := Real.sqrt_pos.2 hxc
    have hepos : 0 < Real.exp (-x)-A := by
      have hqpos : 0 < Real.sqrt (x+c)*(Real.exp (-x)-A) :=
        (by positivity : (0:ℝ) < 1/(2*Real.sqrt M)).trans hq
      exact (mul_pos_iff_of_pos_left hr0).mp hqpos
    have hrs : Real.sqrt (x+c) ≤ Real.sqrt (x+s) := Real.sqrt_le_sqrt (by linarith)
    have hqs : 1/(2*Real.sqrt M) < Real.sqrt (x+s)*(Real.exp (-x)-A) :=
      hq.trans_le (mul_le_mul_of_nonneg_right hrs (le_of_lt hepos))
    have hfrac : (1/(2*Real.sqrt M))/Real.sqrt (x+s) < Real.exp (-x)-A :=
      (div_lt_iff₀ (Real.sqrt_pos.2 hxs)).2 (by simpa only [mul_comm] using hqs)
    have heq : 1/(2*Real.sqrt (x+s))/Real.sqrt M =
        (1/(2*Real.sqrt M))/Real.sqrt (x+s) := by ring
    rw [heq]
    linarith

def threshold (M δ : ℝ) := min
  (Real.exp (-δ)-1/(2*M*Real.sqrt δ))
  (Real.exp (-1)-1/(2*Real.sqrt (M*(1+(M-1)*δ))))

/-- Exactly B(M,delta) from the paper implies decreasing cost for every opponents' sum. -/
theorem manuscript_threshold (A M δ s : ℝ)
    (hA : 0 ≤ A) (hM : 1 ≤ M) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hs : (M-1)*δ ≤ s) (hB : A < threshold M δ) :
    StrictAntiOn (sectionCost A M s) (Icc δ 1) := by
  have hMp : 0 < M := by linarith
  have hMc : 0 ≤ M := le_of_lt hMp
  have hsum : δ+(M-1)*δ = M*δ := by ring
  have hd : 2*Real.sqrt M*Real.sqrt (δ+(M-1)*δ) = 2*M*Real.sqrt δ := by
    rw [hsum, Real.sqrt_mul hMc]
    have hh := Real.sq_sqrt hMc
    nlinarith [congrArg (fun z : ℝ => 2*z*Real.sqrt δ) hh]
  have hd1 : 2*Real.sqrt M*Real.sqrt (1+(M-1)*δ) =
      2*Real.sqrt (M*(1+(M-1)*δ)) := by
    rw [Real.sqrt_mul hMc]
    ring
  apply endpoint_threshold_strict_decrease A M ((M-1)*δ) s δ 1 hA hMp
    (by rw [hsum]; positivity) hδ1 hs
  · rw [hd]
    exact (lt_min_iff.mp hB).1
  · rw [hd1]
    exact (lt_min_iff.mp hB).2


def mean {n : ℕ} (v : Fin (n + 1) → ℝ) : ℝ :=
  (∑ i, v i) / (n + 1 : ℝ)

theorem mean_const (n : ℕ) (a : ℝ) : mean (fun _ : Fin (n + 1) => a) = a := by
  simp only [mean, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, Nat.cast_add, Nat.cast_one]
  apply (div_eq_iff (by positivity : (n : ℝ) + 1 ≠ 0)).2
  ring

theorem mean_add {n : ℕ} (u v : Fin (n + 1) → ℝ) :
    mean (fun i => u i + v i) = mean u + mean v := by
  simp [mean, Finset.sum_add_distrib, add_div]

theorem mean_sub {n : ℕ} (u v : Fin (n + 1) → ℝ) :
    mean (fun i => u i - v i) = mean u - mean v := by
  simp [mean, Finset.sum_sub_distrib, sub_div]

/-- Any control-delta averaging update preserves the initial mean offset. -/
theorem control_offset_invariant {n : ℕ} (s : ℝ) (c d : Fin (n + 1) → ℝ) :
    (s + mean d) - mean (fun i => c i + d i) = s - mean c := by
  rw [mean_add]
  ring

/-- The update separates the selected and unselected control coordinates. -/
theorem control_residual_identity (p c s g : ℝ) :
    c - s + (g + p * (s - c)) = (1 - p) * (c - s) + g := by
  ring

/-- The projected control equals the projected average gradient. -/
theorem projected_control_identity (p c s g : ℝ) (hp : p * p = p) :
    p * (c - s + (g + p * (s - c))) = p * g := by
  calc
    _ = p * g + (p * p - p) * (s - c) := by ring
    _ = _ := by rw [hp]; ring

/-- Balanced controls cancel in the average local direction. -/
theorem centered_controls_cancel {n : ℕ} (g u : Fin (n + 1) → ℝ) :
    mean (fun i => g i + mean u - u i) = mean g := by
  rw [mean_sub, mean_add, mean_const]
  ring

theorem full_correction_mean {n : ℕ} (g c : Fin (n + 1) → ℝ) (s : ℝ) :
    mean (fun i => g i + s - c i) = mean g + (s - mean c) := by
  rw [mean_sub, mean_add, mean_const]
  ring

section PotentialGame

variable {ι : Type*} {α : ι → Type*} [Fintype ι] [DecidableEq ι]

def potential (privateCost : (i : ι) → α i → ℝ) (commonCost : ((i : ι) → α i) → ℝ)
    (x : (i : ι) → α i) : ℝ := (∑ j, privateCost j (x j)) + commonCost x

def stageCost (privateCost : (i : ι) → α i → ℝ) (commonCost : ((i : ι) → α i) → ℝ)
    (constantCost : ι → ℝ) (i : ι) (x : (i : ι) → α i) : ℝ :=
  privateCost i (x i) + commonCost x + constantCost i

omit [DecidableEq ι] in
theorem exact_potential (privateCost : (i : ι) → α i → ℝ) (commonCost : ((i : ι) → α i) → ℝ)
    (constantCost : ι → ℝ) (i : ι) (x y : (i : ι) → α i)
    (hsame : ∀ j, j ≠ i → y j = x j) :
    stageCost privateCost commonCost constantCost i y -
      stageCost privateCost commonCost constantCost i x =
    potential privateCost commonCost y - potential privateCost commonCost x := by
  classical
  have hs : (∑ j, (privateCost j (y j) - privateCost j (x j))) =
      privateCost i (y i) - privateCost i (x i) := by
    apply Finset.sum_eq_single i
    · intro j _ hji
      rw [hsame j hji]
      ring
    · simp
  rw [Finset.sum_sub_distrib] at hs
  unfold stageCost potential
  linarith

def IsNE (cost : ι → ((i : ι) → α i) → ℝ) (x : (i : ι) → α i) : Prop :=
  ∀ i a, cost i x ≤ cost i (Function.update x i a)

theorem finite_exact_potential_has_NE [∀ i, Fintype (α i)] [∀ i, Nonempty (α i)]
    (privateCost : (i : ι) → α i → ℝ) (commonCost : ((i : ι) → α i) → ℝ)
    (constantCost : ι → ℝ) :
    ∃ x, IsNE (stageCost privateCost commonCost constantCost) x := by
  classical
  obtain ⟨x, _, hx⟩ := Finset.exists_min_image (Finset.univ : Finset ((i : ι) → α i))
    (potential privateCost commonCost) Finset.univ_nonempty
  refine ⟨x, ?_⟩
  intro i a
  have hm := hx (Function.update x i a) (Finset.mem_univ _)
  have he := exact_potential privateCost commonCost constantCost i x
    (Function.update x i a) (by intro j hj; simp [Function.update, hj])
  linarith

omit [Fintype ι] in
/-- A strict dominant action at every coordinate gives the unique equilibrium. -/
theorem unique_NE_of_strict_dominance (cost : ι → ((i : ι) → α i) → ℝ)
    (best : (i : ι) → α i)
    (hdom : ∀ (x : (i : ι) → α i) i, x i ≠ best i →
      cost i (Function.update x i (best i)) < cost i x) (x : (i : ι) → α i) :
    IsNE cost x ↔ x = best := by
  constructor
  · intro hx
    funext i
    by_contra hi
    have hstrict := hdom x i hi
    have hne := hx i (best i)
    linarith
  · intro hxb
    subst x
    intro i a
    by_cases ha : a = best i
    · simp [ha]
    · have hstrict := hdom (Function.update best i a) i (by simpa using ha)
      simpa using le_of_lt hstrict

/-- A strictly descending potential path has fewer changes than profiles. -/
theorem strict_path_length_bound {X : Type*} [Fintype X] (length : ℕ)
    (path : Fin (length + 1) → X) (Φ : X → ℝ)
    (hdesc : StrictAnti (fun r => Φ (path r))) : length < Fintype.card X := by
  have hinj : Function.Injective path := by
    intro i j hij
    exact hdesc.injective (congrArg Φ hij)
  have hcard := Fintype.card_le_of_injective path hinj
  simpa using Nat.lt_of_succ_le (by simpa using hcard)

/-- Nested layer selections decrease the actual uncorrected energy. -/
theorem nested_masks_reduce_residual (v : ι → ℝ) (selected larger : Finset ι)
    (hsub : selected ⊆ larger) :
    (∑ j ∈ Finset.univ \ larger, (v j) ^ 2) ≤
      ∑ j ∈ Finset.univ \ selected, (v j) ^ 2 := by
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro j hj
    simp only [Finset.mem_sdiff, Finset.mem_univ, true_and] at hj ⊢
    exact fun hjs => hj (hsub hjs)
  · intro j _ _
    exact sq_nonneg _

omit [Fintype ι] in
/-- Every tail of an action-independent additive stage game is a Nash game.
    The stage cost here is the SAME cost as in the cumulative objective. -/
theorem stage_equilibria_give_tail_equilibria {R : ℕ}
    (cost : Fin R → ι → ((i : ι) → α i) → ℝ) (x : Fin R → (i : ι) → α i)
    (heq : ∀ r, IsNE (cost r) (x r)) (tail : Finset (Fin R))
    (i : ι) (deviation : Fin R → α i) :
    (∑ r ∈ tail, cost r i (x r)) ≤
      ∑ r ∈ tail, cost r i (Function.update (x r) i (deviation r)) := by
  exact Finset.sum_le_sum (fun r _ => heq r i (deviation r))

end PotentialGame
/-- The parameter-free stage cost, on each client's own feasible strategy type. -/
def surrogateCost {n : ℕ} {α : Fin (n+1) → Type*}
    (value : (i : Fin (n+1)) → α i → ℝ) (A : ℝ) (κ : Fin (n+1) → ℝ)
    (i : Fin (n+1)) (x : (j : Fin (n+1)) → α j) : ℝ :=
  Real.exp (-(value i (x i)))+A*(1+value i (x i))+
    Real.sqrt (∑ j, value j (x j))/Real.sqrt (n+1:ℝ)+κ i

/-- The manuscript threshold gives the unique all-corrected Nash equilibrium.
    `value` is injective for suffix selections because all block sizes are positive. -/
theorem manuscript_unique_NE {n : ℕ} {α : Fin (n+1) → Type*}
    (value : (i : Fin (n+1)) → α i → ℝ) (A δ : ℝ) (κ : Fin (n+1) → ℝ)
    (best : (i : Fin (n+1)) → α i)
    (hvalue : ∀ i, Function.Injective (value i))
    (hfeas : ∀ i a, value i a ∈ Icc δ 1)
    (hbest : ∀ i, value i (best i) = 1)
    (hA : 0 ≤ A) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hB : A < threshold (n+1:ℝ) δ) (x : (i : Fin (n+1)) → α i) :
    IsNE (surrogateCost value A κ) x ↔ x = best := by
  apply unique_NE_of_strict_dominance
  intro z i hne
  let s := ∑ j ∈ Finset.univ.erase i, value j (z j)
  have hcard : (Finset.univ.erase i : Finset (Fin (n+1))).card = n := by simp
  have hslow : ((n+1:ℝ)-1)*δ ≤ s := by
    have hh := Finset.sum_le_sum (s := Finset.univ.erase i)
      (fun j _ => (hfeas j (z j)).1)
    simpa only [Finset.sum_const, nsmul_eq_mul, hcard, add_sub_cancel_right] using hh
  have hm : (1:ℝ) ≤ (n+1:ℝ) := by
    have hn : (0:ℝ) ≤ (n:ℝ) := Nat.cast_nonneg n
    linarith
  have hdec := manuscript_threshold A (n+1:ℝ) δ s hA hm hδ hδ1 hslow hB
  have hvlt : value i (z i) < 1 := by
    have hle := (hfeas i (z i)).2
    apply lt_of_le_of_ne hle
    intro heq
    exact hne (hvalue i (heq.trans (hbest i).symm))
  have hcost := hdec (hfeas i (z i)) ⟨hδ1, le_rfl⟩ hvlt
  have split_sum (v : (j : Fin (n+1)) → α j) :
      (∑ j, value j (v j)) = (∑ j ∈ Finset.univ.erase i, value j (v j))+value i (v i) := by
    exact (Finset.sum_erase_add _ _ (Finset.mem_univ i)).symm
  have hsame : (∑ j ∈ Finset.univ.erase i, value j (Function.update z i (best i) j)) = s := by
    apply Finset.sum_congr rfl
    intro j hj
    simp only [Function.update_of_ne (Finset.mem_erase.mp hj).1]
  have hnew : (∑ j, value j (Function.update z i (best i) j)) = s+1 := by
    rw [split_sum, hsame, Function.update_self, hbest]
  have hold : (∑ j, value j (z j)) = s+value i (z i) := split_sum z
  dsimp [surrogateCost]
  rw [hnew, hold, Function.update_self, hbest]
  dsimp [sectionCost] at hcost
  simpa only [add_comm s, add_lt_add_iff_right] using add_lt_add_right hcost (κ i)


def localTrace (η : ℝ) (g q : ℕ → ℝ) : ℕ → ℝ → ℝ
  | 0, w => w
  | k + 1, w => localTrace η g q k w - η * (g k + q k)

theorem local_trace_sum (η w : ℝ) (g q : ℕ → ℝ) (K : ℕ) :
    localTrace η g q K w = w - η * ∑ k ∈ Finset.range K, (g k + q k) := by
  induction K with
  | zero => simp [localTrace]
  | succ K ih => rw [localTrace, ih, Finset.sum_range_succ]; ring

/-- Summing the local steps recovers the control update in Algorithm 1. -/
theorem actual_control_refresh (η w c s : ℝ) (g q : ℕ → ℝ) (K : ℕ)
    (hη : η ≠ 0) (hK : (K : ℝ) ≠ 0) :
    c - s + (w - localTrace η g q K w) / ((K : ℝ) * η) =
      c - s + (∑ k ∈ Finset.range K, g k) / K +
        (∑ k ∈ Finset.range K, q k) / K := by
  rw [local_trace_sum, Finset.sum_add_distrib]
  field_simp
  <;> ring

/-- A fixed binary mask gives the exact refreshed projected gradient. -/
theorem projected_refresh_from_trace (η w c s p : ℝ) (g : ℕ → ℝ) (K : ℕ)
    (hη : η ≠ 0) (hK : (K : ℝ) ≠ 0) (hp : p * p = p) :
    p * (c - s + (w - localTrace η g (fun _ => p * (s - c)) K w) /
      ((K : ℝ) * η)) = p * ((∑ k ∈ Finset.range K, g k) / K) := by
  rw [actual_control_refresh η w c s g (fun _ => p * (s - c)) K hη hK]
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hc : (K : ℝ) * (p * (s - c)) / K = p * (s - c) := by
    field_simp
  rw [hc]
  convert projected_control_identity p c s ((∑ k ∈ Finset.range K, g k) / K) hp using 1 <;> ring

/-- Averaging control increments preserves the initial control-mean offset. -/
theorem control_offset_all_rounds {n : ℕ}
    (s : ℕ → ℝ) (c d : ℕ → Fin (n + 1) → ℝ)
    (hs : ∀ r, s (r + 1) = s r + mean (d r))
    (hc : ∀ r i, c (r + 1) i = c r i + d r i) (r : ℕ) :
    s r - mean (c r) = s 0 - mean (c 0) := by
  induction r with
  | zero => rfl
  | succ r ih =>
    rw [hs r, show c (r + 1) = fun i => c r i + d r i from funext (hc r)]
    rw [control_offset_invariant, ih]


/-- Absorb the displacement term in the finite local-step expansion. -/
theorem local_drift_absorption (t s e B : ℝ)
    (ht : 0 ≤ t) (htmax : t ≤ 1 / 100) (hs : 0 ≤ s) (hsmax : s ≤ 1)
    (he : 0 ≤ e) (hexpand : e ≤ 3 * t ^ 2 * s * e + 3 * t ^ 2 * s * B) :
    e ≤ 6 * t ^ 2 * s * B := by
  have ht2 : t ^ 2 ≤ 1 / 10000 := by nlinarith
  have hq : 3 * t ^ 2 * s ≤ 1 / 2 := by
    have hm := mul_le_mul_of_nonneg_left hsmax (sq_nonneg t)
    nlinarith
  have hm := mul_le_mul_of_nonneg_right hq he
  nlinarith [hexpand, hm]

/-- Refresh bound from smoothness, the noise second moment, and displacement. -/
theorem control_refresh_bound (t e G A V delta Hnext : ℝ)
    (ht : 0 ≤ t) (htmax : t ≤ 1) (he : 0 ≤ e) (hA : 0 ≤ A) (hAV : A ≤ V)
    (hdelta : delta ≤ 3 * t ^ 2 * (G + e + A))
    (hrefresh : Hnext ≤ 4 * e + 4 * delta + 2 * V) :
    Hnext ≤ 16 * e + 12 * t ^ 2 * G + 14 * V := by
  have ht2 : t ^ 2 ≤ 1 := by nlinarith
  have he' := mul_le_mul_of_nonneg_right ht2 he
  have hA' := mul_le_mul_of_nonneg_right ht2 hA
  nlinarith [hdelta, hrefresh, he', hA']

/-- Smoothness and Young's inequality give the one-round optimization bound. -/
theorem smooth_descent_bound (t G e A f fnext : ℝ)
    (ht : 0 ≤ t) (htmax : t ≤ 1 / 6) (hG : 0 ≤ G) (he : 0 ≤ e)
    (hexpand : fnext ≤ f - 3 * t / 4 * G + t * e +
      3 * t ^ 2 / 2 * (G + e + A)) :
    fnext ≤ f - t / 2 * G + 3 / 2 * t * e + 3 / 2 * t ^ 2 * A := by
  have hg := mul_le_mul_of_nonneg_right htmax (show 0 ≤ t * G by positivity)
  have he' := mul_le_mul_of_nonneg_right htmax (show 0 ≤ t * e by positivity)
  nlinarith [hexpand, hg, he']

theorem convex_refresh_bound (e T F V Dnext : ℝ)
    (hgrad : T ≤ 2 * F) (hrefresh : Dnext ≤ 4 * e + 4 * T + 2 * V) :
    Dnext ≤ 4 * e + 8 * F + 2 * V := by linarith

/-- Nonconvex closure: t = beta * eta, s = 1 / rho_g^2.
    e = beta^2 * local drift; H = projected control error;
    G = mean squared gradient; Z = uncorrected heterogeneity;
    V = sigma^2/K; A = sigma^2/(KN).
    f is scaled by beta so the proof is dimensionless. -/
theorem nonconvex_lyapunov
    (t s e H Hnext G Z V A f fnext : ℝ)
    (ht : 0 ≤ t) (htmax : t ≤ 1 / 100)
    (hs : 0 ≤ s) (hsmax : s ≤ 1)
    (hH : 0 ≤ H) (hG : 0 ≤ G)
    (hZ : 0 ≤ Z) (hV : 0 ≤ V)
    (hdrift : e ≤ 6 * t ^ 2 * s * (G + H + Z + V))
    (hcontrol : Hnext ≤ 16 * e + 12 * t ^ 2 * G + 14 * V)
    (hsmooth : fnext ≤ f - t / 2 * G + 3 / 2 * t * e + 3 / 2 * t ^ 2 * A) :
    fnext + 18 * t ^ 3 * s * Hnext ≤
      f + 18 * t ^ 3 * s * H - t / 4 * G +
        t ^ 2 * (3 / 2 * A + 3 * s * V + 1 / 10 * s * Z) := by
  have ht2 : t ^ 2 ≤ 1 / 10000 := by nlinarith
  have hq : 0 ≤ t ^ 2 * s := mul_nonneg (sq_nonneg _) hs
  have hqmax : t ^ 2 * s ≤ 1 / 10000 := by
    have := mul_le_mul_of_nonneg_left hsmax (sq_nonneg t)
    nlinarith
  have hcontrol' := mul_le_mul_of_nonneg_left hcontrol
    (show 0 ≤ 18 * t ^ 3 * s by positivity)
  have hce : 0 ≤ (3 / 2 : ℝ) * t + 288 * t ^ 3 * s := by positivity
  have hd := mul_le_mul_of_nonneg_left hdrift hce
  have hfac : (3 / 2 : ℝ) + 288 * t ^ 2 * s ≤ 5 / 3 := by linarith
  have hsum : 0 ≤ G + H + Z + V := by linarith
  have hf := mul_le_mul_of_nonneg_left hfac
    (show 0 ≤ 6 * t ^ 3 * s * (G + H + Z + V) by positivity)
  have hsmall : 10 * t ^ 2 * s + 216 * (t ^ 2 * s) * t ^ 2 ≤ 1 / 4 := by
    have hp := mul_le_mul_of_nonneg_left ht2 hq
    nlinarith
  have hg := mul_le_mul_of_nonneg_left hsmall (show 0 ≤ t * G by positivity)
  have hh : 0 ≤ 8 * t ^ 3 * s * H := by positivity
  have hzcoef : 10 * t ≤ 1 / 10 := by linarith
  have hz := mul_le_mul_of_nonneg_left hzcoef
    (show 0 ≤ t ^ 2 * s * Z by positivity)
  have hvcoef : 262 * t ≤ 3 := by linarith
  have hv := mul_le_mul_of_nonneg_left hvcoef
    (show 0 ≤ t ^ 2 * s * V by positivity)
  nlinarith [hsmooth, hcontrol', hd, hf, hg, hh, hz, hv]

/-- Convex closure: U = beta^2 E||w-w*||^2 and F = beta E(f(w)-f*).
    D is the actual projected control error at the optimum, not a parameter
    count multiplied into the full control-error energy. -/
theorem convex_lyapunov
    (t s e D Dnext F Z V A U Unext : ℝ)
    (ht : 0 ≤ t) (htmax : t ≤ 1 / 100)
    (hs : 0 ≤ s) (hsmax : s ≤ 1)
    (hD : 0 ≤ D) (hF : 0 ≤ F) (hZ : 0 ≤ Z) (hV : 0 ≤ V)
    (hdrift : e ≤ 6 * t ^ 2 * s * (8 * F + 3 * D + 3 * Z + V))
    (hcontrol : Dnext ≤ 4 * e + 8 * F + 2 * V)
    (hdistance : Unext ≤ U - (2 * t - 6 * t ^ 2) * F +
      t * (1 + 3 * t) * e + 3 * t ^ 2 * A) :
    Unext + t ^ 2 * s * Dnext ≤ U + t ^ 2 * s * D - t * F +
      t ^ 2 * (3 * A + 3 * s * V + s * Z) := by
  have hcontrol' := mul_le_mul_of_nonneg_left hcontrol
    (show 0 ≤ t ^ 2 * s by positivity)
  have hcoeff : 0 ≤ t * (1 + 3 * t) + 4 * t ^ 2 * s := by positivity
  have hd := mul_le_mul_of_nonneg_left hdrift hcoeff
  have hfac : 1 + 3 * t + 4 * t * s ≤ (4 / 3 : ℝ) := by
    have := mul_le_mul_of_nonneg_left hsmax ht
    nlinarith
  have hf := mul_le_mul_of_nonneg_left hfac
    (show 0 ≤ 6 * t ^ 3 * s * (8 * F + 3 * D + 3 * Z + V) by positivity)
  have ht2 : t ^ 2 ≤ 1 / 10000 := by nlinarith
  have hts : t * s ≤ 1 / 100 := by
    have := mul_le_mul_of_nonneg_left hsmax ht
    linarith
  have ht2s : t ^ 2 * s ≤ 1 / 10000 := by
    have := mul_le_mul_of_nonneg_left hsmax (sq_nonneg t)
    linarith
  have hgrad : 6 * t + 8 * t * s + 64 * t ^ 2 * s ≤ 1 := by linarith
  have hg := mul_le_mul_of_nonneg_left hgrad (show 0 ≤ t * F by positivity)
  have h24 : 24 * t ≤ 1 := by linarith
  have hc := mul_le_mul_of_nonneg_left h24 (show 0 ≤ t ^ 2 * s * D by positivity)
  have hz := mul_le_mul_of_nonneg_left h24 (show 0 ≤ t ^ 2 * s * Z by positivity)
  have h8 : 8 * t ≤ 1 := by linarith
  have hv := mul_le_mul_of_nonneg_left h8 (show 0 ≤ t ^ 2 * s * V by positivity)
  nlinarith [hdistance, hcontrol', hd, hf, hg, hc, hz, hv]

/-- Finite telescoping, with no limit interchange and no discarded residual. -/
theorem telescoping (energy loss : ℕ → ℝ) (a b : ℝ) (R : ℕ)
    (hstep : ∀ r < R, energy (r + 1) ≤ energy r - a * loss r + b) :
    a * (∑ r ∈ Finset.range R, loss r) ≤ energy 0 - energy R + R * b := by
  induction R with
  | zero => simp
  | succ R ih =>
    have hprev := ih (fun r hr => hstep r (Nat.lt_trans hr (Nat.lt_succ_self R)))
    have hlast := hstep R (Nat.lt_succ_self R)
    rw [Finset.sum_range_succ]
    push_cast
    nlinarith

/-- A completely explicit tolerance conversion for the nonconvex rate.
    This choice handles B=0 without dividing by the noise level. -/
theorem nonconvex_tolerance
    (β F B ε R η Q : ℝ) (hβ : 0 < β) (hF : 0 ≤ F) (hB : 0 ≤ B)
    (hε : 0 < ε) (hR : 0 < R)
    (hη : η = ε / (β * (100 * ε + 2 * B)))
    (hrounds : 800 * β * F / ε + 16 * β * F * B / ε ^ 2 ≤ R)
    (hbound : Q ≤ 4 * F / (η * R) + β * η * B) :
    0 < η ∧ η ≤ 1 / (100 * β) ∧ Q ≤ ε := by
  have hden : 0 < β * (100 * ε + 2 * B) := by positivity
  have heta : 0 < η := by rw [hη]; exact div_pos hε hden
  have heq : η * (β * (100 * ε + 2 * B)) = ε := by
    rw [hη]
    exact div_mul_cancel₀ ε (ne_of_gt hden)
  have hnoise : β * η * B ≤ ε / 2 := by
    have : 0 ≤ 100 * β * η * ε := by positivity
    nlinarith [heq]
  have hcap : η ≤ 1 / (100 * β) := by
    apply (le_div_iff₀ (by positivity : (0 : ℝ) < 100 * β)).2
    have : 0 ≤ 2 * η * β * B := by positivity
    nlinarith [heq]
  have hrounds' : 8 * F ≤ η * R * ε := by
    have hr := mul_le_mul_of_nonneg_right hrounds (sq_nonneg ε)
    have hε0 : ε ≠ 0 := ne_of_gt hε
    have hrewrite : (800 * β * F / ε + 16 * β * F * B / ε ^ 2) * ε ^ 2 =
        8 * F * (β * (100 * ε + 2 * B)) := by
      field_simp [hε0]
      ring
    rw [hrewrite] at hr
    have hm := mul_le_mul_of_nonneg_left hr (le_of_lt heta)
    have hleft : η * (8 * F * (β * (100 * ε + 2 * B))) = 8 * F * ε := by
      calc
        _ = 8 * F * (η * (β * (100 * ε + 2 * B))) := by ring
        _ = _ := by rw [heq]
    rw [hleft] at hm
    apply (mul_le_mul_right hε).mp
    convert hm using 1 <;> ring
  have hopt : 4 * F / (η * R) ≤ ε / 2 := by
    apply (div_le_iff₀ (mul_pos heta hR)).2
    nlinarith
  exact ⟨heta, hcap, by linarith⟩


/-- The squared residual decreases by exactly the newly corrected block energy. -/
theorem coverage_energy_partition {ι : Type*} [Fintype ι] [DecidableEq ι]
    (v : ι → ℝ) (P Q : Finset ι) (hPQ : P ⊆ Q) :
    (∑ j, if j ∈ P then 0 else (v j)^2) =
      (∑ j, if j ∈ Q then 0 else (v j)^2) +
      ∑ j, if j ∈ Q ∧ j ∉ P then (v j)^2 else 0 := by
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hp : j ∈ P
  · simp [hp, hPQ hp]
  · by_cases hq : j ∈ Q <;> simp [hp, hq]

theorem full_correction_residual_zero {ι : Type*} [Fintype ι] [DecidableEq ι]
    (v : ι → ℝ) :
    (∑ j, if j ∈ (Finset.univ : Finset ι) then (0:ℝ) else (v j)^2) = 0 := by
  simp

/-- e_i is the norm of the initial projected gradient difference.
    Smoothness gives e_i <= beta*d, where d = ||w0-w*||. -/
theorem initial_energy_bound {n : ℕ} (e : Fin (n+1) → ℝ) (β d : ℝ)
    (hβ : 0 ≤ β) (hd : 0 ≤ d) (he : ∀ i, 0 ≤ e i)
    (hsmooth : ∀ i, e i ≤ β*d) :
    mean (fun i => (e i)^2) ≤ β^2*d^2 := by
  rw [← mean_const n (β^2*d^2)]
  apply div_le_div_of_nonneg_right _ (by positivity)
  apply Finset.sum_le_sum
  intro i _
  have := hsmooth i
  have := he i
  have := mul_nonneg hβ hd
  nlinarith

/-- Closed convex recursion, with exact-gradient initialization D0 <= U0.
    Its output is the normalized averaged objective gap from Appendix D. -/
theorem convex_finite_round_bound
    (t s A V Z : ℝ) (U D F e : ℕ → ℝ) (R : ℕ)
    (ht : 0 < t) (htmax : t ≤ 1/100) (hs : 0 ≤ s) (hsmax : s ≤ 1)
    (hV : 0 ≤ V) (hZ : 0 ≤ Z) (hR : 0 < R)
    (hU : ∀ r, 0 ≤ U r) (hD : ∀ r, 0 ≤ D r)
    (hF : ∀ r < R, 0 ≤ F r) (hinit : D 0 ≤ U 0)
    (hdrift : ∀ r < R, e r ≤ 6*t^2*s*(8*F r+3*D r+3*Z+V))
    (hcontrol : ∀ r < R, D (r+1) ≤ 4*e r+8*F r+2*V)
    (hdistance : ∀ r < R, U (r+1) ≤ U r-(2*t-6*t^2)*F r+
      t*(1+3*t)*e r+3*t^2*A) :
    (∑ r ∈ Finset.range R, F r)/(R:ℝ) ≤
      2*U 0/(t*(R:ℝ))+t*(3*A+3*s*V+s*Z) := by
  let C := 3*A+3*s*V+s*Z
  let L := fun r => U r+t^2*s*D r
  have hstep : ∀ r < R, L (r+1) ≤ L r-t*F r+t^2*C := by
    intro r hr
    exact convex_lyapunov t s (e r) (D r) (D (r+1)) (F r) Z V A (U r) (U (r+1))
      (le_of_lt ht) htmax hs hsmax (hD r) (hF r hr) hZ hV
      (hdrift r hr) (hcontrol r hr) (hdistance r hr)
  have htel := telescoping L F t (t^2*C) R hstep
  have hts : t^2*s ≤ 1 := by
    have ht2 : t^2 ≤ 1 := by nlinarith
    have hm := mul_le_mul_of_nonneg_left hsmax (sq_nonneg t)
    nlinarith
  have hinit' : L 0 ≤ 2*U 0 := by
    have h1 := mul_le_mul_of_nonneg_left hinit (show 0 ≤ t^2*s by positivity)
    have h2 := mul_le_mul_of_nonneg_right hts (hU 0)
    dsimp [L]
    linarith
  have hlast : 0 ≤ L R := by
    exact add_nonneg (hU R) (mul_nonneg (by positivity) (hD R))
  have hsum : t*(∑ r ∈ Finset.range R, F r) ≤ 2*U 0+(R:ℝ)*t^2*C := by
    linarith
  have hRreal : 0 < (R:ℝ) := by exact_mod_cast hR
  apply (div_le_iff₀ hRreal).2
  have hright : (2*U 0/(t*(R:ℝ))+t*C)*(R:ℝ) = 2*U 0/t+t*C*(R:ℝ) := by
    field_simp
    ring
  change (∑ r ∈ Finset.range R, F r) ≤ (2*U 0/(t*(R:ℝ))+t*C)*(R:ℝ)
  rw [hright]
  apply (mul_le_mul_left ht).mp
  have heq : t*(2*U 0/t+t*C*(R:ℝ)) = 2*U 0+(R:ℝ)*t^2*C := by
    field_simp
    ring
  rw [heq]
  exact hsum

/-- Closed nonconvex recursion with H0=0, retaining stochastic and residual energies. -/
theorem nonconvex_finite_round_bound
    (t s A V Z floor : ℝ) (f H G e : ℕ → ℝ) (R : ℕ)
    (ht : 0 < t) (htmax : t ≤ 1/100) (hs : 0 ≤ s) (hsmax : s ≤ 1)
    (hV : 0 ≤ V) (hZ : 0 ≤ Z) (hR : 0 < R)
    (hH : ∀ r, 0 ≤ H r) (hG : ∀ r < R, 0 ≤ G r)
    (hinit : H 0 = 0) (hlower : floor ≤ f R)
    (hdrift : ∀ r < R, e r ≤ 6*t^2*s*(G r+H r+Z+V))
    (hcontrol : ∀ r < R, H (r+1) ≤ 16*e r+12*t^2*G r+14*V)
    (hsmooth : ∀ r < R, f (r+1) ≤ f r-t/2*G r+3/2*t*e r+3/2*t^2*A) :
    (∑ r ∈ Finset.range R, G r)/(R:ℝ) ≤
      4*(f 0-floor)/(t*(R:ℝ))+t*(6*A+12*s*V+2/5*s*Z) := by
  let C := 3/2*A+3*s*V+1/10*s*Z
  let L := fun r => f r+18*t^3*s*H r
  have hstep : ∀ r < R, L (r+1) ≤ L r-t/4*G r+t^2*C := by
    intro r hr
    exact nonconvex_lyapunov t s (e r) (H r) (H (r+1)) (G r) Z V A (f r) (f (r+1))
      (le_of_lt ht) htmax hs hsmax (hH r) (hG r hr) hZ hV
      (hdrift r hr) (hcontrol r hr) (hsmooth r hr)
  have htel := telescoping L G (t/4) (t^2*C) R hstep
  have hstart : L 0 = f 0 := by simp [L, hinit]
  have hlast : floor ≤ L R := by
    have hHR := hH R
    have hp : 0 ≤ 18*t^3*s*H R := by positivity
    dsimp [L]
    linarith
  rw [hstart] at htel
  have hsum : t*(∑ r ∈ Finset.range R, G r) ≤
      4*(f 0-floor)+(R:ℝ)*t^2*(6*A+12*s*V+2/5*s*Z) := by
    have hraw : t/4*(∑ r ∈ Finset.range R, G r) ≤ f 0-floor+(R:ℝ)*(t^2*C) := by
      exact htel.trans (add_le_add_right (sub_le_sub_left hlast (f 0)) _)
    have hm := mul_le_mul_of_nonneg_left hraw (by norm_num : (0:ℝ) ≤ 4)
    convert hm using 1 <;> dsimp [C] <;> ring
  have hRreal : 0 < (R:ℝ) := by exact_mod_cast hR
  apply (div_le_iff₀ hRreal).2
  apply (mul_le_mul_left ht).mp
  have heq : t*((4*(f 0-floor)/(t*(R:ℝ))+t*(6*A+12*s*V+2/5*s*Z))*(R:ℝ)) =
      4*(f 0-floor)+(R:ℝ)*t^2*(6*A+12*s*V+2/5*s*Z) := by
    field_simp
    ring
  rw [heq]
  exact hsum

/-- Explicit step size for the sharper convex rate in the revised appendix. -/
theorem convex_tolerance_exact_initialization
    (β W C ε R η Q : ℝ) (hβ : 0 < β) (hW : 0 ≤ W)
    (hC : 0 ≤ C) (hε : 0 < ε) (hR : 0 < R)
    (hη : η = 1/(100*β+2*C/ε))
    (hrounds : 4*W*(100*β+2*C/ε)/ε ≤ R)
    (hbound : Q ≤ 2*W/(η*R)+η*C) :
    0 < η ∧ η ≤ 1/(100*β) ∧ Q ≤ ε := by
  have hden : 0 < 100*β+2*C/ε := by positivity
  have heta : 0 < η := by rw [hη]; positivity
  have heq : η*(100*β+2*C/ε) = 1 := by rw [hη]; field_simp
  have hnoisePart : 0 ≤ η*(2*C/ε) := by positivity
  have hβPart : 0 ≤ η*(100*β) := by positivity
  have hcap : η ≤ 1/(100*β) := by
    apply (le_div_iff₀ (by positivity : (0:ℝ) < 100*β)).2
    nlinarith [heq]
  have hnoise : η*C ≤ ε/2 := by
    have hn : η*(2*C/ε) ≤ 1 := by nlinarith [heq]
    have hm := mul_le_mul_of_nonneg_right hn (le_of_lt hε)
    have hre : η*(2*C/ε)*ε = 2*η*C := by field_simp; ring
    rw [hre] at hm
    linarith
  have hr : 4*W ≤ η*R*ε := by
    have hb := (div_le_iff₀ hε).mp hrounds
    have hm := mul_le_mul_of_nonneg_left hb (le_of_lt heta)
    have heqW := congrArg (fun z : ℝ => 4*W*z) heq
    nlinarith only [hm, heqW]
  have hopt : 2*W/(η*R) ≤ ε/2 := by
    apply (div_le_iff₀ (mul_pos heta hR)).2
    nlinarith
  exact ⟨heta, hcap, by linarith⟩

/-- The direct gradient-ratio reconstruction error, coordinate by coordinate. -/
theorem reconstruction_error (z gw gb dw db : ℝ)
    (hchain : gw = gb*z) (hden : gb+db ≠ 0) :
    (gw+dw)/(gb+db)-z = (dw-z*db)/(gb+db) := by
  rw [hchain]
  field_simp
  ring

/-- Sum of coordinate squares is the Euclidean reconstruction-error energy. -/
theorem reconstruction_energy {ι : Type*} [Fintype ι]
    (z gw dw : ι → ℝ) (gb db : ℝ)
    (hchain : ∀ j, gw j = gb*z j) (hden : gb+db ≠ 0) :
    (∑ j, ((gw j+dw j)/(gb+db)-z j)^2) =
      (∑ j, (dw j-z j*db)^2)/(gb+db)^2 := by
  simp_rw [reconstruction_error _ _ _ _ _ (hchain _) hden, div_pow]
  rw [Finset.sum_div]

end AdapCTAFormal

#print axioms AdapCTAFormal.h_deriv
#print axioms AdapCTAFormal.q_deriv
#print axioms AdapCTAFormal.h_nonpos_high
#print axioms AdapCTAFormal.h_strictAnti_low
#print axioms AdapCTAFormal.q_endpoint_min
#print axioms AdapCTAFormal.sectionCost_deriv
#print axioms AdapCTAFormal.endpoint_threshold_strict_decrease
#print axioms AdapCTAFormal.manuscript_threshold
#print axioms AdapCTAFormal.mean_const
#print axioms AdapCTAFormal.mean_add
#print axioms AdapCTAFormal.mean_sub
#print axioms AdapCTAFormal.control_offset_invariant
#print axioms AdapCTAFormal.control_residual_identity
#print axioms AdapCTAFormal.projected_control_identity
#print axioms AdapCTAFormal.centered_controls_cancel
#print axioms AdapCTAFormal.full_correction_mean
#print axioms AdapCTAFormal.exact_potential
#print axioms AdapCTAFormal.finite_exact_potential_has_NE
#print axioms AdapCTAFormal.unique_NE_of_strict_dominance
#print axioms AdapCTAFormal.strict_path_length_bound
#print axioms AdapCTAFormal.nested_masks_reduce_residual
#print axioms AdapCTAFormal.stage_equilibria_give_tail_equilibria
#print axioms AdapCTAFormal.manuscript_unique_NE
#print axioms AdapCTAFormal.local_trace_sum
#print axioms AdapCTAFormal.actual_control_refresh
#print axioms AdapCTAFormal.projected_refresh_from_trace
#print axioms AdapCTAFormal.control_offset_all_rounds
#print axioms AdapCTAFormal.local_drift_absorption
#print axioms AdapCTAFormal.control_refresh_bound
#print axioms AdapCTAFormal.smooth_descent_bound
#print axioms AdapCTAFormal.convex_refresh_bound
#print axioms AdapCTAFormal.nonconvex_lyapunov
#print axioms AdapCTAFormal.convex_lyapunov
#print axioms AdapCTAFormal.telescoping
#print axioms AdapCTAFormal.nonconvex_tolerance
#print axioms AdapCTAFormal.coverage_energy_partition
#print axioms AdapCTAFormal.full_correction_residual_zero
#print axioms AdapCTAFormal.initial_energy_bound
#print axioms AdapCTAFormal.convex_finite_round_bound
#print axioms AdapCTAFormal.nonconvex_finite_round_bound
#print axioms AdapCTAFormal.convex_tolerance_exact_initialization
#print axioms AdapCTAFormal.reconstruction_error
#print axioms AdapCTAFormal.reconstruction_energy
