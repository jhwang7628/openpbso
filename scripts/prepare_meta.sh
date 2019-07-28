for p in *; do
    if [ -d $p ]; then
        if [ "$(ls -A ${p}/ffat_map-fdtd)" ]; then # check if empty
        # if [ -d $p/ffat_map-fdtd ]; then
            mat=$(ls ${p}/*.txt);
            echo "$p : ${mat}";
            echo "`pwd`/$p/$p.tet.obj\n`pwd`/$p/${p}_surf.modes\n`pwd`/${mat}\n`pwd`/$p/ffat_map-fdtd" > ../../assets/meta/10k_2/$p.meta;
        fi;
    fi;
done
