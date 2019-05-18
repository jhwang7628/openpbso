#include "igl/serialize.h"
int main() {
#if 0
    bool b = true;
    unsigned int num = 10;
    std::vector<float> vec = {0.1,0.002,5.3};

    // use overwrite = true for the first serialization to create or overwrite an
    // existing file
    igl::serialize(b,"B","filename",true);
    // append following serialization to existing file
    igl::serialize(num,"Number","filename");
    igl::serialize(vec,"VectorName","filename");
#else
    bool b;
    unsigned int num;
    std::vector<float> vec;

    // use overwrite = true for the first serialization to create or overwrite an
    // existing file
    //igl::serialize(b,"B","filename",true);
    //// append following serialization to existing file
    //igl::serialize(num,"Number","filename");
    //igl::serialize(vec,"VectorName","filename");

    // deserialize back to variables
    igl::deserialize(b,"B","filename");
    igl::deserialize(num,"Number","filename");
    igl::deserialize(vec,"VectorName","filename");
#endif
    std::cout << b << std::endl;
    std::cout << num << std::endl;
    std::cout << vec.at(1) << std::endl;
    return 0;

}
