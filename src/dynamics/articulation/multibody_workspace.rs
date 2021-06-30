use crate::dynamics::RigidBodyVelocity;
use crate::math::Real;
use na::DVector;

/// A temporary workspace for various updates of the multibody.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub(crate) struct MultibodyWorkspace {
    pub accs: Vec<RigidBodyVelocity>,
    pub ndofs_vec: DVector<Real>,
}

impl MultibodyWorkspace {
    /// Create an empty workspace.
    pub fn new() -> Self {
        MultibodyWorkspace {
            accs: Vec::new(),
            ndofs_vec: DVector::zeros(0),
        }
    }

    /// Resize the workspace so it is enough for `nlinks` links.
    pub fn resize(&mut self, nlinks: usize, ndofs: usize) {
        self.accs.resize(nlinks, RigidBodyVelocity::zero());
        self.ndofs_vec = DVector::zeros(ndofs)
    }
}

// pub(crate) struct SolverWorkspace<N: RealField, Handle: BodyHandle, CollHandle: ColliderHandle> {
//     jacobians: DVector<N>,
//     constraints: ConstraintSet<N, Handle, CollHandle, usize>,
// }
//
// impl<N: RealField, Handle: BodyHandle, CollHandle: ColliderHandle>
//     SolverWorkspace<N, Handle, CollHandle>
// {
//     pub fn new() -> Self {
//         SolverWorkspace {
//             jacobians: DVector::zeros(0),
//             constraints: ConstraintSet::new(),
//         }
//     }
//
//     pub fn resize(&mut self, nconstraints: usize, ndofs: usize) {
//         let j_len = nconstraints * ndofs * 2;
//
//         if self.jacobians.len() != j_len {
//             self.jacobians = DVector::zeros(j_len);
//         }
//     }
// }
