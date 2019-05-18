#include "ffat_solver.h"
int main() {
    typedef double T;
    typedef Gpu_Wavesolver::FFAT_Map<T,3> FFAT_Map;
    std::unique_ptr<std::map<int,FFAT_Map>> ffat_maps(
        FFAT_Map::LoadAll("../data/wine/ffat_maps")
    );
    FFAT_Map map0 = ffat_maps->at(0);
#define COUT_SDUMP(x) std::cout << "x = " << x << std::endl;
    COUT_SDUMP(map0.GetCenter());
}
