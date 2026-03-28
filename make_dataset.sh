/workspace/bullet/target/release/bullet-utils viribinpack interleave \
  hobbes-6.vf hobbes-7.vf hobbes-8.vf hobbes-9.vf hobbes-10.vf hobbes-11.vf hobbes-12.vf hobbes-13.vf hobbes-15.vf hobbes-16.vf hobbes-19.vf hobbes-20.vf hobbes-21.vf hobbes-23.vf hobbes-25.vf hobbes-27.vf hobbes-31-dfrc.vf hobbes-31-std.vf hobbes-33-dfrc.vf hobbes-36-20k.vf hobbes-36.vf hobbes-37-20k.vf hobbes-38.vf hobbes-38-2.vf \
  --output hobbes-s1.vf

rm hobbes-6.vf hobbes-7.vf hobbes-8.vf hobbes-9.vf hobbes-10.vf hobbes-11.vf

/workspace/bullet/target/release/bullet-utils viribinpack interleave \
  hobbes-12.vf hobbes-13.vf hobbes-15.vf hobbes-16.vf hobbes-19.vf hobbes-20.vf hobbes-21.vf hobbes-23.vf hobbes-25.vf hobbes-27.vf hobbes-31-dfrc.vf hobbes-31-std.vf hobbes-33-dfrc.vf hobbes-36-20k.vf hobbes-36.vf hobbes-37-20k.vf hobbes-38.vf hobbes-38-2.vf \
  --output hobbes-s2.vf

rm hobbes-12.vf hobbes-13.vf hobbes-15.vf

/workspace/bullet/target/release/bullet-utils viribinpack interleave \
  hobbes-16.vf hobbes-19.vf hobbes-20.vf hobbes-21.vf hobbes-23.vf hobbes-25.vf hobbes-27.vf hobbes-31-dfrc.vf hobbes-31-std.vf hobbes-33-dfrc.vf hobbes-36-20k.vf hobbes-36.vf hobbes-37-20k.vf hobbes-38.vf hobbes-38-2.vf \
  --output hobbes-s3.vf

rm hobbes-16.vf hobbes-19.vf hobbes-20.vf

/workspace/bullet/target/release/bullet-utils viribinpack interleave \
  hobbes-21.vf hobbes-23.vf hobbes-25.vf hobbes-27.vf hobbes-31-dfrc.vf hobbes-31-std.vf hobbes-33-dfrc.vf hobbes-36-20k.vf hobbes-36.vf hobbes-37-20k.vf hobbes-38.vf hobbes-38-2.vf \
  --output hobbes-s4.vf

rm hobbes-21.vf hobbes-23.vf

/workspace/bullet/target/release/bullet-utils viribinpack interleave \
  hobbes-25.vf hobbes-27.vf hobbes-31-dfrc.vf hobbes-31-std.vf hobbes-33-dfrc.vf hobbes-36-20k.vf hobbes-36.vf hobbes-37-20k.vf hobbes-38.vf hobbes-38-2.vf \
  --output hobbes-s5.vf

rm hobbes-25.vf hobbes-27.vf hobbes-31-dfrc.vf hobbes-31-std.vf hobbes-33-dfrc.vf hobbes-36-20k.vf hobbes-36.vf hobbes-37-20k.vf hobbes-38.vf hobbes-38-2.vf