#!/bin/bash
wget https://cdn.kernel.org/pub/linux/kernel/v4.x/linux-4.20.tar.xz
tar -xJf linux-4.20.tar.xz
tar -cf linux-4.20.tar linux-4.20
rm -r linux-4.20
rm linux-4.20.tar.xz

wget http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip
mkdir silesia
mv silesia.zip silesia
cd silesia
unzip silesia.zip
rm silesia.zip
cd ..
tar -cf silesia.tar silesia
rm -r silesia

