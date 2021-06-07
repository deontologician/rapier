use na::{
    self, DVectorSliceMut, Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3, VectorSlice3,
};

use crate::dynamics::{Articulation, IntegrationParameters, RigidBodyVelocity};
use crate::math::{JacobianSliceMut, Real};
use crate::utils::WCrossMatrix;

/// A articulation that allows only all rotational degrees of freedom between two multibody links.
#[derive(Copy, Clone, Debug)]
pub struct BallArticulation {
    rot: UnitQuaternion<Real>,

    jacobian_v: Matrix3<Real>,
    jacobian_dot_v: Matrix3<Real>,
}

impl BallArticulation {
    /// Create a ball articulation with the an initial position given by a rotation in axis-angle form.
    pub fn new(axisangle: Vector3<Real>) -> Self {
        BallArticulation {
            rot: UnitQuaternion::new(axisangle),
            jacobian_v: na::zero(),
            jacobian_dot_v: na::zero(),
        }
    }
}

impl Articulation for BallArticulation {
    #[inline]
    fn ndofs(&self) -> usize {
        3
    }

    fn body_to_parent(
        &self,
        parent_shift: &Vector3<Real>,
        body_shift: &Vector3<Real>,
    ) -> Isometry3<Real> {
        let trans = Translation3::from(parent_shift - self.rot * body_shift);
        Isometry3::from_parts(trans, self.rot)
    }

    fn update_jacobians(&mut self, body_shift: &Vector3<Real>, vels: &[Real]) {
        let shift = self.rot * -body_shift;
        let angvel = VectorSlice3::from_slice(vels);

        self.jacobian_v = shift.gcross_matrix().transpose();
        self.jacobian_dot_v = angvel.cross(&shift).gcross_matrix().transpose();
    }

    fn jacobian(&self, transform: &Isometry3<Real>, out: &mut JacobianSliceMut<Real>) {
        // FIXME: could we avoid the computation of rotation matrix on each `jacobian_*`  ?
        let rotmat = transform.rotation.to_rotation_matrix();
        out.fixed_rows_mut::<3>(0)
            .copy_from(&(rotmat * self.jacobian_v));
        out.fixed_rows_mut::<3>(3).copy_from(rotmat.matrix());
    }

    fn jacobian_dot(&self, transform: &Isometry3<Real>, out: &mut JacobianSliceMut<Real>) {
        let rotmat = transform.rotation.to_rotation_matrix();
        out.fixed_rows_mut::<3>(0)
            .copy_from(&(rotmat * self.jacobian_dot_v));
    }

    fn jacobian_dot_veldiff_mul_coordinates(
        &self,
        transform: &Isometry3<Real>,
        acc: &[Real],
        out: &mut JacobianSliceMut<Real>,
    ) {
        let angvel = Vector3::from_row_slice(&acc[..3]);
        let rotmat = transform.rotation.to_rotation_matrix();
        let res = rotmat * angvel.gcross_matrix() * self.jacobian_v;
        out.fixed_rows_mut::<3>(0).copy_from(&res);
    }

    fn jacobian_mul_coordinates(&self, acc: &[Real]) -> RigidBodyVelocity {
        let angvel = Vector3::from_row_slice(&acc[..3]);
        let linvel = self.jacobian_v * angvel;
        RigidBodyVelocity::new(linvel, angvel)
    }

    fn jacobian_dot_mul_coordinates(&self, acc: &[Real]) -> RigidBodyVelocity {
        let angvel = Vector3::from_row_slice(&acc[..3]);
        let linvel = self.jacobian_dot_v * angvel;
        RigidBodyVelocity::new(linvel, na::zero())
    }

    fn default_damping(&self, out: &mut DVectorSliceMut<Real>) {
        out.fill(na::convert(0.1f64))
    }

    fn integrate(&mut self, parameters: &IntegrationParameters, vels: &[Real]) {
        let angvel = Vector3::from_row_slice(&vels[..3]);
        let disp = UnitQuaternion::new_eps(angvel * parameters.dt, 0.0);
        self.rot = disp * self.rot;
    }

    fn apply_displacement(&mut self, disp: &[Real]) {
        let angle = Vector3::from_row_slice(&disp[..3]);
        let disp = UnitQuaternion::new(angle);
        self.rot = disp * self.rot;
    }

    #[inline]
    fn clone(&self) -> Box<dyn Articulation> {
        Box::new(*self)
    }
}
