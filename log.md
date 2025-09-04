▌It seems that the MoE + ICM did not do what I expect. Maybe the primitives is
▌wrong in my code.For example, For loco I should also set destination x and y
▌besides delta x and delta y. Also, it seems that we cannot carry out the expert
▌per frame. we should at least wait for the previous expert to finish the
▌command and then proceed to the next gate right? like when we output a
▌destination or hop command we should at least wait for it to finish the whole
▌movement and then proceed to the next. since we use merge (top k) we should
▌maybe consider a asynchronous step like things for this MoE gating? What do you
▌think? Do not modify code just chat

> You’re hitting a real limitation of per-frame MoE blending: it’s great for
  smooth interpolation but poor for long-horizon, discrete skills that must
  “run to completion” (e.g., hop, backflip, kick). Blending top‑k experts each
  step cancels intentions and rarely allows a primitive to finish. Also, some
  primitives likely need absolute goals (destination x,y) rather than deltas to
  make progress measurable.
  
  What’s going wrong
  
  - Step-wise mixing: Top‑k averaging each frame destroys the commitment needed
  for options (skills) to complete. A hop or “go to (x,y)” gets interrupted
  before it can finish.
  - Command spec mismatch: Loco with only delta velocities lacks a goal-to-go;
  the policy can dither or hover. Absolute targets (x,y) or “distance-to-go”
  help.
  - Sticky isn’t enough: Stickiness=0.85 still updates every step. Without
  explicit option durations/termination, the gate flip-flops.
  - Expert cadence: Some primitives are designed to operate over a horizon
  (contact cycles, airtime), not frame-by-frame request/response.
  
  What to do (no code, design adjustments)
  
  - Define option semantics per primitive
      - Loco: target position (x,y) or path segment + optional target yaw;
  termination when within tolerance or timeout.
      - Hop: apex height target + flight detection; termination when landing +
  stance stable.
      - Swing/contact: duration or event-based completion (e.g., foot makes/
  breaks contact).
      - Pose: desired pitch/roll/height with tolerance + min hold time.
  - Enforce commitment (semi‑MDP style)
      - Manager (gate) chooses one expert + its command arguments and commits
  for N steps or until termination criteria met, then may switch.
      - While committed: freeze gate to that expert (k=1), keep command constant
  (or time-scheduled), and ignore per-frame re-gating.
      - Record (expert, args, duration) segments for PPO/ICM credit assignment
  at boundaries (approximate SMDP if you keep PPO).
  - Separate command generation from action mixing
      - Use the gate to choose the expert and generate the command (goal/
  params), not to blend actions.
      - Let the chosen expert produce actions autonomously for its horizon;
  merging actions (top‑k) only makes sense for micro-refinements, not macro
  skills.
  - Add temporal “done” signals
      - Termination predicates per primitive: position tolerance, contact events
  (takeoff/landing), max duration, velocity thresholds.
      - A simple min-duration (e.g., 0.4–0.8 s) plus a done condition prevents
  chattering and allows skill completion.
  - Enhance commands for locomotive goals
      - Include absolute (x,y) or a local waypoint; feed error-to-go in
  observations so the primitive can converge rather than drift.
      - Deltas alone encourage incremental, possibly oscillatory behavior;
  destination creates a finish line for the option.
  - Keep ICM as the exploration driver
      - Don’t add shaping penalties to “keep safe” during training; instead,
  ensure the option scheduler lets risky skills finish.
      - ICM will push to novel expert sequences; the commitment mechanism
  ensures those sequences can actually manifest as full motions.
  
  Debug/validation checks (to confirm the diagnosis)
  
  - Is the gate changing every frame? If yes, you’re likely interrupting skills
  prematurely.
  - Do expert outputs (means) differ meaningfully across experts? If masks/
  experts are misaligned or identical, the gate won’t explore.
  - Does loco without absolute targets make progress to any point in space? If
  not, commanders lack a solvable objective.
  
  Transitional compromise (if you don’t want full SMDP yet)
  
  - Gate every K steps: only allow gate updates every fixed chunk (e.g., 10–20
  timesteps). This simulates commitment and often unlocks hop/pose skills.
  - Use top‑1 during commitment: switch to k=1 when committed; only at decision
  boundaries compute top‑k for selection.
  
  Bottom line
  
  - Blending actions per-frame is the wrong tool for skills that require
  completion. Move to “select‑and‑commit” semantics (option scheduling), use
  goal-centric commands for locomotion, and define simple termination criteria.
  Keep ICM as-is — it will then discover challenging sequences of completed
  skills rather than jittery mixtures that never finish.

▌but i need MERGE! for example what if i want both walking at the same time
▌swing one of its foot for kicking the ball? Also what if I want backflip that
▌can only achieve by several combiination of the primtiives?

> You’re right to require MERGE. You want real compositional control: keep a
  base behavior (walk), overlay a limb swing (kick), and sometimes chain/overlap
  multiple primitives (backflip phases). Per‑frame convex mixing of all experts
  won’t cut it; you need structured, asynchronous composition with commitments
  and clear interfaces.
  
  What to change conceptually (no code, design only)
  
  - Split roles: base vs overlay
      - Base expert: provides whole‑body locomotion/posture (e.g., walking).
  Runs with a commitment window (e.g., 0.3–1.0 s) so it finishes contact phases.
      - Overlay expert(s): inject task‑specific changes on a subset of DOFs or
  in task space (e.g., one leg swing for a kick). Shorter commitments (e.g.,
  0.15–0.3 s), can start/stop asynchronously.
      - Overlay expert(s): inject task‑specific changes on a subset of DOFs or
  in task space (e.g., one leg swing for a kick). Shorter commitments (e.g.,
  0.15–0.3 s), can start/stop asynchronously.
  - 
  Two gates, asynchronous
      - Base gate: selects exactly one base expert + parameters; commits for
  N_base steps or until termination criteria (stance cycle boundary, tolerance
  met).
      - Overlay gate: selects zero or one overlay expert (or top‑k overlays
  if compatible) + parameters; commits for N_overlay steps; can start mid‑base
  window and end independently.
      - This preserves MERGE: base continues while overlay modifies part of
  the body.
  - 
  Where to merge (not naive torque mixing)
      - Masked DOF blending: action = base_action + M_overlay ⊙ overlay_residual
  (M_overlay masks affected DOFs/EE). Keeps base stable while allowing local
  modulation.
      - Task‑space residuals for overlay: prefer Cartesian swing/impact targets
  for the end‑effector; convert to joint residuals via IK/impedance. This
  composes better than summing torques.
      - Priority blending: base in priority 0; overlay in priority 1; if
  conflicts, clip overlay to keep stability (e.g., null‑space projection or
  weighted least‑squares with base constraints).
  - 
  Commit semantics and phase awareness
      - Base experts expose a contact/phase variable; commit until a safe switch
  point (e.g., footfall boundary), not every frame.
      - Overlay experts define preconditions (e.g., selected foot must be in
  swing), duration (min–max), and termination (target velocity/position reached
  or timeout).
      - For backflip‑like motions: model them as macro‑primitives with internal
  phases (crouch → thrust → tuck → untuck → land), chosen by the base gate with
  a full‑sequence commitment; overlays can decorate edges (e.g., arm balance).
  - 
  Command interface clean‑up
      - Locomotion: add absolute (x,y) or short‑horizon waypoint; delta‑only
  leads to dithering. Provide a distance‑to‑goal in obs so the base expert can
  “finish” something.
      - Overlay (kick): params like EE target velocity at contact, intercept
  timing, foot selection. Use ball rel pos/vel from obs to time activation.
      - Hop/backflip: params for apex height, angular impulse, tuck duration;
  the expert owns its internal phase variable.
  - 
  Top‑k merging reinterpreted
      - Top‑k within each role: base_top1 (commit) + overlay_topk (if overlays
  act on disjoint DOFs or can be summed as residuals).
      - Gate weights act as residual gains for overlays (scale the residual, not
  a global action average).
  - 
  Safety without reward shaping
      - No new rewards. Enforce feasibility in the merger: mask DOFs, respect
  joint/torque limits, contact‑aware activation (don’t swing a stance foot), and
  clip overlay residuals under high base effort.
      - Keep ICM to push the scheduler to try novel base/overlay sequences; the
  commitment logic lets them actually complete.
  - 
  For rare combos (e.g., backflip)
      - Provide a macro option for “flip” with an internal controller/phase;
  the gate commits to it. Overlays can bias arms/legs but do not interrupt the
  macro’s backbone.
      - Allow overlapping windows: e.g., initiate flip (base), add brief overlay
  to increase angular momentum, then release.
  
  Concrete next steps (implementation plan, when you’re ready)
  
  - Define expert metadata: role (base/overlay), DOF mask or task‑space target,
  min/max duration, preconditions, termination.
  - Add a tiny scheduler:
      - Base scheduler: pick base expert + args; lock for N_base or phase done.
      - Overlay scheduler: can trigger independently (subject to preconditions);
  lock for N_overlay.
  - Change the merge operator:
      - Compute base_action (full DOFs).
      - Compute overlay_residual(s) in masked DOFs or task space, then compose:
  action = base_action + sum_i M_i ⊙ residual_i. Enforce constraints.
  - Expand commands minimally:
      - Base: add (x,y) waypoint support (or short‑horizon local frame target).
      - Overlay: add EE swing targets/timing.
  
  This keeps MERGE (true concurrency), preserves the ability to discover
  multi‑primitive combos, and avoids per‑frame averaging that cancels
  intentions. It also fits your “ICM‑only” philosophy: exploration decides
  what sequences/overlays to try; commitment and structured blending make them
  realizable.


