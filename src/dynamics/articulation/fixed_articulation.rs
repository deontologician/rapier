use na::DVectorSliceMut;

use crate::dynamics::Articulation;

use crate::dynamics::{IntegrationParameters, RigidBodyVelocity};
use crate::math::{Isometry, JacobianSliceMut, Real, Translation, Vector};

/// A articulation that does not allow any relative degrees of freedom.
#[derive(Copy, Clone, Debug)]
pub struct FixedArticulation {
    body_to_parent: Isometry<Real>,
}

impl FixedArticulation {
    /// Create a articulation that does not a allow any degrees of freedom between two multibody links.
    ///
    /// The descendent attached to this articulation will have a position maintained to `pos_wrt_body`
    /// relative to its parent.
    pub fn new(pos_wrt_body: Isometry<Real>) -> Self {
        FixedArticulation {
            body_to_parent: pos_wrt_body.inverse(),
        }
    }
}

impl Articulation for FixedArticulation {
    fn ndofs(&self) -> usize {
        0
    }

    fn body_to_parent(
        &self,
        parent_shift: &Vector<Real>,
        body_shift: &Vector<Real>,
    ) -> Isometry<Real> {
        let parent_trans = Translation::from(*parent_shift);
        let body_trans = Translation::from(*body_shift);
        parent_trans * self.body_to_parent * body_trans
    }

    fn update_jacobians(&mut self, _: &Vector<Real>, _: &[Real]) {}

    fn jacobian(&self, _: &Isometry<Real>, _: &mut JacobianSliceMut<Real>) {}

    fn jacobian_dot(&self, _: &Isometry<Real>, _: &mut JacobianSliceMut<Real>) {}

    fn jacobian_dot_veldiff_mul_coordinates(
        &self,
        _: &Isometry<Real>,
        _: &[Real],
        _: &mut JacobianSliceMut<Real>,
    ) {
    }

    fn integrate(&mut self, _: Real, _: &[Real]) {}
    fn apply_displacement(&mut self, _: &[Real]) {}

    fn jacobian_mul_coordinates(&self, _: &[Real]) -> RigidBodyVelocity {
        RigidBodyVelocity::zero()
    }

    fn jacobian_dot_mul_coordinates(&self, _: &[Real]) -> RigidBodyVelocity {
        RigidBodyVelocity::zero()
    }

    fn default_damping(&self, _: &mut DVectorSliceMut<Real>) {}

    #[inline]
    fn clone(&self) -> Box<dyn Articulation> {
        Box::new(*self)
    }
}
