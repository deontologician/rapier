use std::ops::{Deref, DerefMut};

use crate::dynamics::{Articulation, RigidBodyHandle};
use crate::math::{AngularInertia, Isometry, Point, Real, Vector};
use crate::prelude::RigidBodyVelocity;

/// One link of a multibody.
pub struct MultibodyLink {
    pub(crate) name: String,
    // FIXME: make all those private.
    pub(crate) internal_id: usize,
    pub(crate) assembly_id: usize,
    pub(crate) impulse_id: usize,
    pub(crate) is_leaf: bool,

    // XXX: rename to "articulation".
    // (And rename the full-coordinates articulation constraints ArticulationConstraint).
    pub(crate) parent_internal_id: usize,
    pub(crate) articulation: Box<dyn Articulation>,
    pub(crate) parent_shift: Vector<Real>,
    pub(crate) body_shift: Vector<Real>,

    // Change at each time step.
    pub(crate) parent_to_world: Isometry<Real>,
    // TODO: should this be removed in favor of the rigid-body position?
    pub(crate) local_to_world: Isometry<Real>,
    pub(crate) local_to_parent: Isometry<Real>,
    // FIXME: put this on a workspace buffer instead ?
    pub(crate) velocity_dot_wrt_joint: RigidBodyVelocity,
    // J' q' in world space. FIXME: what could be a better name ?
    pub(crate) velocity_wrt_joint: RigidBodyVelocity,
    // J  q' in world space is the rigid-body velocity.
    pub(crate) rigid_body: RigidBodyHandle,
}

impl MultibodyLink {
    /// Creates a new multibody link.
    pub fn new(
        rigid_body: RigidBodyHandle,
        internal_id: usize,
        assembly_id: usize,
        impulse_id: usize,
        parent_internal_id: usize,
        articulation: Box<dyn Articulation>,
        parent_shift: Vector<Real>,
        body_shift: Vector<Real>,
        parent_to_world: Isometry<Real>,
        local_to_world: Isometry<Real>,
        local_to_parent: Isometry<Real>,
    ) -> Self {
        let is_leaf = true;
        let velocity_dot_wrt_joint = RigidBodyVelocity::zero();
        let velocity_wrt_joint = RigidBodyVelocity::zero();

        MultibodyLink {
            name: String::new(),
            internal_id,
            assembly_id,
            impulse_id,
            is_leaf,
            parent_internal_id,
            articulation,
            parent_shift,
            body_shift,
            parent_to_world,
            local_to_world,
            local_to_parent,
            velocity_dot_wrt_joint,
            velocity_wrt_joint,
            rigid_body,
        }
    }

    /// Checks if this link is the root of the multibody.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.internal_id == 0
    }

    /// Reference to the articulation attaching this link to its parent.
    #[inline]
    pub fn articulation(&self) -> &dyn Articulation {
        &*self.articulation
    }

    /// Mutable reference to the articulation attaching this link to its parent.
    #[inline]
    pub fn articulation_mut(&mut self) -> &mut dyn Articulation {
        &mut *self.articulation
    }

    /// The shift between this link's parent and this link articulation origin.
    #[inline]
    pub fn parent_shift(&self) -> &Vector<Real> {
        &self.parent_shift
    }

    /// The shift between this link's articulation origin and this link origin.
    #[inline]
    pub fn body_shift(&self) -> &Vector<Real> {
        &self.body_shift
    }

    /// This link's name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets this link's name.
    #[inline]
    pub fn set_name(&mut self, name: String) {
        self.name = name
    }

    /// The handle of this multibody link.
    #[inline]
    pub fn link_id(&self) -> usize {
        self.internal_id
    }

    /// The handle of the parent link.
    #[inline]
    pub fn parent_id(&self) -> Option<usize> {
        if self.internal_id != 0 {
            Some(self.parent_internal_id)
        } else {
            None
        }
    }
}

// FIXME: keep this even if we already have the Index2 traits?
pub(crate) struct MultibodyLinkVec(pub Vec<MultibodyLink>);

impl MultibodyLinkVec {
    #[inline]
    pub fn get_mut_with_parent(&mut self, i: usize) -> (&mut MultibodyLink, &MultibodyLink) {
        let parent_id = self[i].parent_internal_id;

        assert!(
            parent_id != i,
            "Internal error: circular rigid body dependency."
        );
        assert!(parent_id < self.len(), "Invalid parent index.");

        unsafe {
            let rb = &mut *(self.get_unchecked_mut(i) as *mut _);
            let parent_rb = &*(self.get_unchecked(parent_id) as *const _);
            (rb, parent_rb)
        }
    }
}

impl Deref for MultibodyLinkVec {
    type Target = Vec<MultibodyLink>;

    #[inline]
    fn deref(&self) -> &Vec<MultibodyLink> {
        let MultibodyLinkVec(ref me) = *self;
        me
    }
}

impl DerefMut for MultibodyLinkVec {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec<MultibodyLink> {
        let MultibodyLinkVec(ref mut me) = *self;
        me
    }
}
