use na::{self, RealField, U3, VectorSlice3, Vector, Point, Matrix3, Isometry, Translation3, UnitQuaternion};

use crate::utils::GeneralizedCross;
use crate::dynamics::Articulation;
use crate::solver::IntegrationParameters;
use crate::math::{Velocity, JacobianSliceMut};


#[derive(Copy, Clone, Debug)]
pub struct SlotArticulation {
    shift: Vector<Real>,
    axis:  Unit<Vector<Real>>,

    pos: Real,
    rot: Rotation<Real>,

    jacobian_v:     Matrix3<Real>,
    jacobian_dot_v: Matrix3<Real>,
}

impl SlotArticulation {
    pub fn new(center: Point<Real>, axisangle: Vector<Real>) -> Self {
        SlotArticulation {
            shift:          -center.coords,
            rot:            UnitQuaternion::new(axisangle),
            jacobian_v:     na::zero(),
            jacobian_dot_v: na::zero()
        }
    }
}


impl<Real: RealField, Handle: BodyHandle> Articulation<Real, Handle> for SlotArticulation {
    #[inline]
    fn ndofs(&self) -> usize {
        ANGULAR_DIM + 1
    }

    fn body_to_parent(&self) -> Isometry<Real> {
        let trans = Translation3::from(self.shift + &*self.axis * self.pos);
        let rot   = 
        Isometry::from_parts(trans, self.rot)
    }

    fn update_jacobians(&mut self, vels: &[Real]) {
        let parent_shift = self.rot * self.shift;
        let angvel       = VectorSlice3::new(vels);

        self.jacobian_v     = parent_shift.gcross_matrix_tr();
        self.jacobian_dot_v = angvel.cross(&parent_shift).gcross_matrix_tr();
    }

    fn jacobian(&self, transform: &Isometry<Real>, out: &mut JacobianSliceMut<Real>) {
        // FIXME: could we avoid the computation of rotation matrix on each `jacobian_*`  ?
        let rotmat = transform.rotation.to_rotation_matrix();
        out.fixed_rows_mut::<U3>(0).copy_from(&(rotmat * self.jacobian_v));
        out.fixed_rows_mut::<U3>(3).copy_from(rotmat.matrix());
    }

    fn jacobian_dot(&self, transform: &Isometry<Real>, out: &mut JacobianSliceMut<Real>) {
        let rotmat = transform.rotation.to_rotation_matrix();
        out.fixed_rows_mut::<U3>(0).copy_from(&(rotmat * self.jacobian_dot_v));
    }

    fn jacobian_dot_veldiff_mul_coordinates(&self, transform: &Isometry<Real>, acc: &[Real],
                                            out: &mut JacobianSliceMut<Real>) {
        let angvel = Vector::from_row_slice(&acc[.. 3]);
        let rotmat = transform.rotation.to_rotation_matrix();
        let res    = rotmat * angvel.gcross_matrix() * self.jacobian_v;
        out.fixed_rows_mut::<U3>(0).copy_from(&res);
    }

    fn jacobian_mul_coordinates(&self, acc: &[Real]) -> Velocity<Real> {
        let angvel = Vector::from_row_slice(&acc[.. 3]);
        let linvel = self.jacobian_v * angvel;
        Velocity::new(linvel, angvel)
    }

    fn jacobian_dot_mul_coordinates(&self, acc: &[Real]) -> Velocity<Real> {
        let angvel = Vector::from_row_slice(&acc[.. 3]);
        let linvel = self.jacobian_dot_v * angvel;
        Velocity::new(linvel, na::zero())
    }

    fn integrate(&mut self, dt: Real, vels: &[Real]) {
        let angvel = Vector::from_row_slice(&vels[.. 3]);
        let disp   = UnitQuaternion::new(angvel *  dt);
        self.rot   = disp * self.rot;
    }
}
