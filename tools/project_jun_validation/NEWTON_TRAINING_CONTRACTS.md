# Recovered Newton run configuration comparison

The final-run W&B configuration exports are preserved privately. Their SHA-256 values are `265264afc4b727988c23162348df98f0768b4f29149f594bbfe60d402db175ab` for `kyfj909z` and `1eb76c8ae751895003ccad5198cabb1fe5779546252383980527b43ea1981352` for `p24gbesn`. These are serialized configurations; dictionary order does not prove runtime observation concatenation or exact executed source bytes.

The `kyfj909z` configuration names `g1_29dof_soft_contact_newton.mdp.observations:velocity_commands_wxyz` in the policy group and specifies a 10-step flattened policy history. Its runner lists the actor input as the policy group. This is evidence of an intended wxyz conversion, not recovery of the named historical function's implementation. The reconstructed live evaluator uses a separately recovered candidate task and has not established source equivalence.

The `p24gbesn` export instead lists a separate command group using `isaaclab.envs.mdp.observations:generated_commands`; its actor network lists inputs `policy` and `command`. The 937-dimensional reconstructed actor concatenates 930+7 and must remain separate from the 1,000-dimensional `kyfj909z` contract. Both exports specify 5 ms simulation timestep, decimation four, and 20-second episodes, but nominal fields do not prove equivalent runtime physics.

The exported reward definitions also differ. For example, `kyfj909z` records -1 for vertical velocity and -0.05 for angular xy velocity, while `p24gbesn` records -0.25 for each; only the latter lists a nonzero stance-foot-angle penalty. These differences make their trained-policy comparison unsuitable as an isolated backend comparison. They do not establish the cause of either policy's behavior.

The raw W&B exports, checkpoints and traces are not included in this public branch. The private local archive and PACE files hold them; published hashes identify the observations above without claiming historical source recovery.
