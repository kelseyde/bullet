#!/bin/sh
set -e

BULLET=/workspace/bullet/target/release/bullet-utils

OLD="hobbes-6.vf hobbes-7.vf hobbes-8.vf hobbes-9.vf hobbes-10.vf
     hobbes-11.vf hobbes-12.vf hobbes-13.vf hobbes-15.vf hobbes-16.vf
     hobbes-19.vf hobbes-20.vf hobbes-21.vf hobbes-23.vf hobbes-25.vf
     hobbes-27.vf hobbes-31-dfrc.vf hobbes-31-std.vf hobbes-33-dfrc.vf
     hobbes-36-20k.vf hobbes-36.vf hobbes-37-20k.vf hobbes-38.vf
     hobbes-38-2.vf hobbes-42.vf"

BEST="hobbes-42-12k-dfrc.vf hobbes-42-12k-std-2.vf hobbes-42-12k-std-3.vf hobbes-46-12k.vf"

$BULLET viribinpack interleave $OLD $BEST --output hobbes-all.vf
rm $OLD

$BULLET viribinpack interleave $BEST --output hobbes-best.vf
rm $BEST