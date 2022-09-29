//#include "sequential.h"
//namespace rcsc {
//Sequential::Sequential()
//{

//}

//int Sequential::add(Layer *layer)
//{
//    graphs.push_back(*layer);
//    std::vector<Matrix> layer_params = layer->get_params();
//    params.push_back(layer_params);
//    return 0;
//}

//int Sequential::forward(Matrix &input)
//{
//    Matrix layer_input = input;
//    for (auto &layer:graphs)
//    {
//        layer_input = layer.forward(layer_input);
//    }
//    return 0;
//}

//int Sequential::backward(Matrix &grad)
//{
//    Matrix layer_grad = grad;
//    for (int i=graphs.size()-1; i > 0; i--)
//    {
//        auto &layer = graphs[i];
//        layer_grad = layer.backward(layer_grad);
//    }
//    return 0;
//}

//}

