#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>
#include <array>
#include <type_traits>
#include <cmath>

template<typename T_type> class tensor;

template<typename T_type>
class tensor{
    public:
        unsigned int count = 1;
        std::vector<unsigned int> strides;
        std::vector<unsigned int> dims;
        std::vector<T_type> flat_tensor;
        T_type* flat_tensor_ptr;
        size_t offset = 0;

        template<typename InpType>
        tensor(const InpType& inp_tensor);

        template<typename Other_type>
        tensor(const tensor<Other_type>& other_tensor);

        tensor(std::vector<unsigned int> given_dims, std::vector<T_type> given_flat_tensor, T_type* given_flat_tensor_ptr, size_t given_offset = 0);

        tensor(std::vector<unsigned int> given_dims, std::vector<T_type> given_flat_tensor); 

        void operator+=(const auto& scalar);

        template<typename Other_type>
        void operator+=(const tensor<Other_type>& inp_tensor);

        tensor operator+(const auto& scalar);

        template<typename Other_type>
        tensor operator+(const tensor<Other_type>& inp_tensor);

        void operator-=(const auto& scalar);

        template<typename Other_type>
        void operator-=(const tensor<Other_type>& inp_tensor);
        
        tensor operator-(const auto& scalar); 

        template<typename Other_type>
        tensor operator-(const tensor<Other_type>& inp_tensor);

        void operator*=(const auto& scalar);   

        template<typename Other_type>
        void operator*=(const tensor<Other_type>& inp_tensor);    

        tensor operator*(const auto& scalar);

        template<typename Other_type>
        tensor operator*(const tensor<Other_type>& inp_tensor);   

        void operator/=(const auto& scalar);   

        template<typename Other_type>
        void operator/=(const tensor<Other_type>& inp_tensor);

        tensor operator/(const auto& scalar);      

        template<typename Other_type>
        tensor operator/(const tensor<Other_type>& inp_tensor);

        void pow(const auto& exponent);

        template<typename Other_type>
        void pow(const tensor<Other_type> tensor_exponent);

        tensor operator[](int start = 0, std::optional<int> finish = std::nullopt, int step = 1);

        template<typename Other_type>
        void operator=(tensor<Other_type> inp_tensor);

        void operator=(const auto& scalar);

        template<typename ... Args>
        tensor& reshape(Args...args);
        
        tensor& T();
        
        T_type* get_ptr(size_t index);

        tensor copy();

    private:
        void calc_count_of_elements();

        template<typename Current_type>
        void extract_dim(const Current_type& current_dim);
        
        template<typename Current_type>
        void extract_flat_data(const Current_type& current_tensor);

        void calc_strides();

};  

template<typename T_type>
template<typename InpType>
tensor<T_type>::tensor(const InpType& inp_tensor) {
    extract_dim(inp_tensor);
    calc_count_of_elements();
    flat_tensor.reserve(count);
    extract_flat_data(inp_tensor);     
    calc_strides();
    flat_tensor_ptr = &flat_tensor[0];
}

template<typename T_type>
template<typename Other_type>
tensor<T_type>::tensor(const tensor<Other_type>& other_tensor) {
    this -> dims = other_tensor.dims;
    this -> count = other_tensor.count;
    this -> strides = other_tensor.strides;
    this -> flat_tensor.reserve(other_tensor.count);
    this -> flat_tensor.assign(other_tensor.flat_tensor.begin(), other_tensor.flat_tensor.end());
}

template<typename T_type>
tensor<T_type>::tensor(std::vector<unsigned int> given_dims, std::vector<T_type> given_flat_tensor, T_type* given_flat_tensor_ptr, size_t given_offset): dims(std::move(given_dims)), flat_tensor(std::move(given_flat_tensor)), flat_tensor_ptr(std::move(given_flat_tensor_ptr)), offset(std::move(given_offset)) {
    calc_count_of_elements();
    calc_strides();
    flat_tensor_ptr = &flat_tensor[0];
}

template<typename T_type>
tensor<T_type>::tensor(std::vector<unsigned int> given_dims, std::vector<T_type> given_flat_tensor): dims(given_dims), flat_tensor(given_flat_tensor) {
    calc_count_of_elements();
    calc_strides();
    flat_tensor_ptr = &flat_tensor[0];
}


template<typename T_type>
void tensor<T_type>::operator+=(const auto& scalar) {
    for(int i = 0; i<count; i++){
        *(this->flat_tensor_ptr + (offset + i)) += scalar;
    }
}

template<typename T_type>
template<typename Other_type>
void tensor<T_type>::operator+=(const tensor<Other_type>& inp_tensor) {
    for(int i = 0; i<count; i++){
        *(this->flat_tensor_ptr + (offset + i)) += inp_tensor.flat_tensor[i];
    }
}

template<typename T_type>
tensor<T_type> tensor<T_type>::operator+(const auto& scalar) {
    tensor result(*this);
    result += scalar;
    return result;
}


template<typename T_type>
template<typename Other_type>
tensor<T_type> tensor<T_type>::operator+(const tensor<Other_type>& inp_tensor) {
    tensor result(*this);
    result += inp_tensor;
    return result;
}

template<typename T_type>
void tensor<T_type>::operator-=(const auto& scalar) {
    for(int i = 0; i<count; i++){
        *(this->flat_tensor_ptr + (offset + i)) -= scalar;
    }
}

template<typename T_type>
template<typename Other_type>
void tensor<T_type>::operator-=(const tensor<Other_type>& inp_tensor) {
    for(int i = 0; i<count; i++){
        *(this->flat_tensor_ptr + (offset + i)) -= inp_tensor.flat_tensor[i];
    }
}

template<typename T_type>
tensor<T_type> tensor<T_type>::operator-(const auto& scalar) {
    tensor result(*this);
    result -= scalar;
    return result;
}

template<typename T_type>
template<typename Other_type>
tensor<T_type> tensor<T_type>::operator-(const tensor<Other_type>& inp_tensor) {
    tensor result(*this);
    result -= inp_tensor;
    return result;
}

template<typename T_type>
void tensor<T_type>::operator*=(const auto& scalar) {
    for(int i = 0; i<count; i++){
        *(this->flat_tensor_ptr + (offset + i)) *= scalar;
    }
}

template<typename T_type>
template<typename Other_type>
void tensor<T_type>::operator*=(const tensor<Other_type>& inp_tensor) {
    for(int i = 0; i<count; i++){
        *(this->flat_tensor_ptr + (offset + i)) *= inp_tensor.flat_tensor[i];
    }
}
   
template<typename T_type>
tensor<T_type> tensor<T_type>::operator*(const auto& scalar) {
    tensor result(*this);
    result *= scalar;
    return result;
}

template<typename T_type>
template<typename Other_type>
tensor<T_type> tensor<T_type>::operator*(const tensor<Other_type>& inp_tensor) {
    tensor result(*this);
    result *= inp_tensor;
    return result;
}
 
template<typename T_type>
void tensor<T_type>::operator/=(const auto& scalar) {
    for(int i = 0; i<count; i++){
        *(this->flat_tensor_ptr + (offset + i)) /= scalar;
    }
}
    
template<typename T_type>
template<typename Other_type>
void tensor<T_type>::operator/=(const tensor<Other_type>& inp_tensor) {
    for(int i = 0; i<count; i++){
        *(this->flat_tensor_ptr + (offset + i)) /= inp_tensor.flat_tensor[i];
    }
}

template<typename T_type>
tensor<T_type> tensor<T_type>::operator/(const auto& scalar) {
    tensor result(*this);
    result /= scalar;
    return result;
}

template<typename T_type>
template<typename Other_type>
tensor<T_type> tensor<T_type>::operator/(const tensor<Other_type>& inp_tensor) {
    tensor result(*this);
    result /= inp_tensor;
    return result;
}

template<typename T_type>
void tensor<T_type>::pow(const auto& exponent) {
    for(unsigned int i = 0; i < size(this->flat_tensor); i++) {
        *(this->flat_tensor_ptr + (offset + i)) = std::pow((this->flat_tensor[i]), exponent); 
    }
}

template<typename T_type>
template<typename Other_type>
void tensor<T_type>::pow(const tensor<Other_type> tensor_exponent) {
    for(unsigned int i = 0; i < size(this); i++) {
        *(this->flat_tensor_ptr + (offset + i)).pow(tensor_exponent[i]);
    }
}


template<typename T_type>
tensor<T_type> tensor<T_type>::operator[](int start, std::optional<int> finish, int step) {
    int final_finish = finish.value_or(start + 1);
    
    std::vector<unsigned int> out_dims {dims};
    out_dims[0] = (final_finish - start + step - 1) / step;

    std::vector<T_type> out_flat_tensor;
    out_flat_tensor.resize(out_dims[0] * strides[0]);

    if(step != 1){
        auto iter = out_flat_tensor.begin();
        for(int i = start; i < final_finish; i += step) {
            auto src_start = flat_tensor.begin() + (i * strides[0]);
            std::copy(src_start, src_start + strides[0], iter);

            iter += strides[0];
        }
    }
    else {
        auto src_start = flat_tensor.begin() + start * strides[0];
        std::copy(src_start, src_start + out_dims[0] * strides[0], out_flat_tensor.begin());
    }

    if(out_dims[0] == 1 and size(out_dims) != 1) {
        out_dims.erase(out_dims.begin());
    }

    return tensor<T_type> (std::move(out_dims), std::move(out_flat_tensor), std::move(flat_tensor_ptr), std::move(offset + start));
}

template<typename T_type>
template<typename Other_type>
void tensor<T_type>::operator=(tensor<Other_type> inp_tensor) {
    std::copy(inp_tensor.begin(), inp_tensor.end(), flat_tensor_ptr);
}

template<typename T_type>
void tensor<T_type>::operator=(const auto& scalar) {
    *this = tensor<T_type> (std::vector<unsigned int> (1), std::vector<T_type> (1));
}


template<typename T_type>
template<typename ... Args>
tensor<T_type>& tensor<T_type>::reshape(Args...args) {
    dims = std::vector<unsigned int> {static_cast<unsigned int>(args) ...};
    strides.clear();
    calc_strides();
    return *this;
}

template<typename T_type>
tensor<T_type>& tensor<T_type>::T() {
    std::reverse(dims.begin(), dims.end());
    strides.clear();
    calc_strides();
    return *this;
}

template<typename T_type>
void tensor<T_type>::calc_count_of_elements(){
    count = 1;
    for(unsigned int d: dims){
        count *= d;
    }
    return;
}

template<typename T_type>
template<typename Current_type>
void tensor<T_type>::extract_dim(const Current_type& current_dim){
    if constexpr (std::is_arithmetic_v<Current_type>) {
        if (dims.empty()) {
            dims.push_back(1);
        }
        return;
    }
    else{
        dims.push_back(size(current_dim));
        if(size(current_dim)>0){extract_dim(current_dim[0]);}
    }
}

template<typename T_type>
template<typename Current_type>
void tensor<T_type>::extract_flat_data(const Current_type& current_tensor){
    if constexpr (std::is_arithmetic_v<Current_type>) {
        flat_tensor.push_back(current_tensor);
        return;
    }
    else{
        for(auto& t: current_tensor){
            extract_flat_data(t);
        }
    }
}

template<typename T_type>
void tensor<T_type>::calc_strides(){
    strides.resize(size(dims));
    int current_stride = 1;
    for(int i = size(dims)-1; i >= 0; i--){
        strides[i] = current_stride;
        current_stride *= dims[i];
    }
}

template<typename T_type>
T_type* tensor<T_type>::get_ptr(size_t index) {
    return flat_tensor_ptr += offset + index;
}

template<typename T_type>
tensor<T_type> tensor<T_type>::copy() {
    tensor<T_type> tensor_copy (dims ,flat_tensor);
    return tensor_copy;
}


int main(){
    std::array<std::array<float, 3>, 2> a {{{1,2,3}, {4,5,6},}};
    std::array<float, 1> b {42};
    tensor<float> rtg {b};
    tensor<float> t{a};
    for(float i: t.flat_tensor){std::cout<<i<<" ";}
    std::cout<<std::endl;
    for(float i: t.dims){std::cout<<i<<" ";}
    std::cout<<std::endl;
    for(float i: t.strides){std::cout<<i<<" ";}
    std::cout<<std::endl;


    t[0][0] = 10;
    for(float i: t.flat_tensor){std::cout<<i<<" ";}
    std::cout<<std::endl;
    for(float i: t.dims){std::cout<<i<<" ";}
    std::cout<<std::endl;
    for(float i: t.strides){std::cout<<i<<" ";}
    std::cout<<std::endl;
}