use gpurs_artifact::SharedMemory;
use rustc_data_structures::fx::FxHashMap;

use crate::{
    codegen::ir::{Module, Type, Value},
    stable::Instance,
};

const STATIC_LIMIT: u64 = 48 * 1024;

pub struct SharedAllocations {
    pub required: SharedMemory,
    slots: FxHashMap<(Instance, usize, Vec<u64>), (Value, u64)>,
}

impl Default for SharedAllocations {
    fn default() -> Self {
        SharedAllocations {
            required: SharedMemory {
                static_bytes: 8,
                dynamic_bytes: 0,
            },
            slots: FxHashMap::default(),
        }
    }
}

impl SharedAllocations {
    pub fn allocate(
        &mut self,
        module: &mut Module,
        site: (Instance, usize, Vec<u64>),
        size: u64,
        align: u64,
    ) -> (Value, u64) {
        self.slots
            .entry(site)
            .or_insert_with(|| {
                let start = self.required.static_bytes.next_multiple_of(align);
                if start + size <= STATIC_LIMIT {
                    self.required.static_bytes = start + size;
                    (
                        module.global(
                            "gpurs.shared",
                            3,
                            false,
                            Value::undef(Type::bytes(size)),
                            align,
                        ),
                        0,
                    )
                } else {
                    let offset = self.required.dynamic_bytes.next_multiple_of(align);
                    self.required.dynamic_bytes = offset + size;
                    (
                        module.external("gpurs_dynamic", Type::bytes(0), 3, align),
                        offset,
                    )
                }
            })
            .clone()
    }
}
