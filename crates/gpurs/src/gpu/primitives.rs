//! The `primitives!` declaration macro for inline-PTX primitives.
//!
//! A block opens with default definition fields, then lists items as
//! `[unsafe][(overrides)][targets] TEXT CONSTRAINTS [pub(..)] fn name(..) [-> result];`.
//! Items are public unless `pub(..)` restricts them, and `pub(self)` keeps them private.
//! Overrides are definition fields, optionally led by the `warp` or `block` convergence
//! shorthand; targets are `sm_90a | sm_100a` or `>= sm_90`; TEXT is a literal or `concat!(..)`.

macro_rules! primitives {
    ($($key:ident = $value:tt),* $(,)? ; $($items:tt)*) => {
        primitives!(@attributes [$($key = $value,)*] [] $($items)*);
    };
    (@attributes $header:tt []) => {};
    (@attributes $header:tt [$($attribute:tt)*] #[$($meta:tt)*] $($rest:tt)*) => {
        primitives!(@attributes $header [$($attribute)* #[$($meta)*]] $($rest)*);
    };
    (@attributes $header:tt $attributes:tt unsafe $($rest:tt)*) => {
        primitives!(@overrides $header $attributes [unsafe] $($rest)*);
    };
    (@attributes $header:tt $attributes:tt $($rest:tt)*) => {
        primitives!(@overrides $header $attributes [] $($rest)*);
    };
    (@overrides $header:tt $attributes:tt $safety:tt (warp $(, $($field:tt)*)?) $($rest:tt)*) => {
        primitives!(@targets $header $attributes $safety [convergence = Warp, $($($field)*)?] $($rest)*);
    };
    (@overrides $header:tt $attributes:tt $safety:tt (block $(, $($field:tt)*)?) $($rest:tt)*) => {
        primitives!(@targets $header $attributes $safety [convergence = Block, $($($field)*)?] $($rest)*);
    };
    (@overrides $header:tt $attributes:tt $safety:tt ($($field:tt)*) $($rest:tt)*) => {
        primitives!(@targets $header $attributes $safety [$($field)*] $($rest)*);
    };
    (@overrides $header:tt $attributes:tt $safety:tt $($rest:tt)*) => {
        primitives!(@targets $header $attributes $safety [] $($rest)*);
    };
    (@targets $header:tt $attributes:tt $safety:tt [$($field:tt)*] >= $target:ident $($rest:tt)*) => {
        primitives!(@text $header $attributes $safety [targets = stringify!(>= $target), $($field)*] $($rest)*);
    };
    (@targets $header:tt $attributes:tt $safety:tt [$($field:tt)*]
        $target:ident $(| $alternative:ident)* $text:literal $($rest:tt)*) => {
        primitives!(@text $header $attributes $safety
            [targets = stringify!($target $(| $alternative)*), $($field)*] $text $($rest)*);
    };
    (@targets $header:tt $attributes:tt $safety:tt [$($field:tt)*]
        $target:ident $(| $alternative:ident)* concat!($($part:tt)*) $($rest:tt)*) => {
        primitives!(@text $header $attributes $safety
            [targets = stringify!($target $(| $alternative)*), $($field)*] concat!($($part)*) $($rest)*);
    };
    (@targets $header:tt $attributes:tt $safety:tt $fields:tt $($rest:tt)*) => {
        primitives!(@text $header $attributes $safety $fields $($rest)*);
    };
    (@text $header:tt $attributes:tt $safety:tt $fields:tt
        concat!($($part:tt)*) $constraints:literal $($rest:tt)*) => {
        primitives!(@item $header $attributes $safety $fields [concat!($($part)*)] $constraints $($rest)*);
    };
    (@text $header:tt $attributes:tt $safety:tt $fields:tt $text:literal $constraints:literal $($rest:tt)*) => {
        primitives!(@item $header $attributes $safety $fields [$text] $constraints $($rest)*);
    };
    (@item $header:tt $attributes:tt $safety:tt $fields:tt $text:tt
        $constraints:literal fn $($rest:tt)*) => {
        primitives!(@emit $header $attributes $safety $fields $text $constraints [pub] fn $($rest)*);
    };
    (@item $header:tt $attributes:tt $safety:tt $fields:tt $text:tt
        $constraints:literal pub(self) fn $($rest:tt)*) => {
        primitives!(@emit $header $attributes $safety $fields $text $constraints [] fn $($rest)*);
    };
    (@item $header:tt $attributes:tt $safety:tt $fields:tt $text:tt
        $constraints:literal pub($($scope:tt)*) fn $($rest:tt)*) => {
        primitives!(@emit $header $attributes $safety $fields $text $constraints
            [pub($($scope)*)] fn $($rest)*);
    };
    (@emit [$($header:tt)*] [$($attribute:tt)*] [$($safety:tt)?] [$($field:tt)*] [$($text:tt)*]
        $constraints:literal [$($visibility:tt)*] fn $name:ident($($arguments:tt)*)
        $(-> $result:ty)?; $($rest:tt)*) => {
        #[gpurs_macros::primitive(
            $($header)*
            lowering = Assembly { text: $($text)*, constraints: $constraints },
            $($field)*
        )]
        $($attribute)*
        $($visibility)* $($safety)? fn $name($($arguments)*) $(-> $result)?;
        primitives!(@attributes [$($header)*] [] $($rest)*);
    };
}
