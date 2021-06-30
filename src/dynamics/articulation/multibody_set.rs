use crate::data::{Arena, Coarena};
use crate::dynamics::{Multibody, RigidBodyHandle};
use crate::parry::partitioning::IndexedData;
use std::ops::Index;

/*
 * FIXME: this is not the way we should represent multibodies.
 *        Instead, we should have a graph, just like the jointSet,
 *        and the connected components of that graph implicitly
 *        represent multibodies.
 * FIXME: once we switch to the graph representation, we should rename
 *        this `ArticulationSet` and `ArticulationHandle`.
 */

/// The unique handle of a multibody added to a `MultibodySet`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct MultibodyHandle(pub crate::data::arena::Index);

impl MultibodyHandle {
    /// Converts this handle into its (index, generation) components.
    pub fn into_raw_parts(self) -> (u32, u32) {
        self.0.into_raw_parts()
    }

    /// Reconstructs an handle from its (index, generation) components.
    pub fn from_raw_parts(id: u32, generation: u32) -> Self {
        Self(crate::data::arena::Index::from_raw_parts(id, generation))
    }

    /// An always-invalid rigid-body handle.
    pub fn invalid() -> Self {
        Self(crate::data::arena::Index::from_raw_parts(
            crate::INVALID_U32,
            crate::INVALID_U32,
        ))
    }
}

impl Default for MultibodyHandle {
    fn default() -> Self {
        Self::invalid()
    }
}

impl IndexedData for MultibodyHandle {
    fn default() -> Self {
        Self(IndexedData::default())
    }

    fn index(&self) -> usize {
        self.0.index()
    }
}

/// A set of rigid bodies that can be handled by a physics pipeline.
pub struct MultibodySet {
    pub(crate) multibodies: Arena<Multibody>,
    pub(crate) rb2mb: Coarena<(MultibodyHandle, usize)>,
}

impl MultibodySet {
    pub fn new() -> Self {
        Self {
            multibodies: Arena::new(),
            rb2mb: Coarena::new(),
        }
    }

    // FIXME: we should insert articulations instead.
    pub fn insert(&mut self, multibody: Multibody) -> MultibodyHandle {
        let handle = self.multibodies.insert(multibody);
        let multibody = &self.multibodies[handle];
        for (link_id, link) in multibody.links().enumerate() {
            self.rb2mb
                .insert(link.rigid_body.0, (MultibodyHandle(handle), link_id));
        }

        MultibodyHandle(handle)
    }

    pub fn rigid_body_link(&self, rb: RigidBodyHandle) -> Option<&(MultibodyHandle, usize)> {
        self.rb2mb.get(rb.0)
    }
}

impl Index<MultibodyHandle> for MultibodySet {
    type Output = Multibody;

    fn index(&self, index: MultibodyHandle) -> &Multibody {
        &self.multibodies[index.0]
    }
}
