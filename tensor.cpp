#pragma GCC optimize("O3")

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

inline constexpr std::nullopt_t _ = std::nullopt;

int sgn(const auto& val) {
    return (val < 0) ? -1 : 1;
}


template<typename T_type> class tensor;

template<typename T_type>
class tensor{
    public:
        int count = 1;
        std::vector<int> strides;
        std::vector<int> dims;
        std::vector<T_type> flat_tensor;
        T_type* flat_tensor_ptr;
        size_t offset = 0;

        template<typename InpType>
        tensor(const InpType& inp_tensor);

        template<typename Other_type>
        tensor(const tensor<Other_type>& other_tensor);

        tensor(std::vector<int> given_dims, std::vector<T_type> given_flat_tensor, T_type* given_flat_tensor_ptr, std::vector<int> given_strides, size_t given_offset = 0);

        tensor(std::vector<int> given_dims, std::vector<T_type> given_flat_tensor); 

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

        template<typename Other_type>
        tensor dot(tensor<Other_type>& inp_tensor);

        tensor operator[](std::optional<int> start, std::optional<int> finish, std::optional<int> step);

        tensor operator[](int index);

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
tensor<T_type>::tensor(std::vector<int> given_dims, std::vector<T_type> given_flat_tensor, T_type* given_flat_tensor_ptr, std::vector<int> given_strides, size_t given_offset): dims((given_dims)), flat_tensor((given_flat_tensor)), flat_tensor_ptr((given_flat_tensor_ptr)), strides(given_strides), offset((given_offset)) {
    calc_count_of_elements();
}

template<typename T_type>
tensor<T_type>::tensor(std::vector<int> given_dims, std::vector<T_type> given_flat_tensor): dims(given_dims), flat_tensor(given_flat_tensor) {
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
    for(int i = 0; i < size(this->flat_tensor); i++) {
        *(this->flat_tensor_ptr + (offset + i)) = std::pow((this->flat_tensor[i]), exponent); 
    }
}

template<typename T_type>
template<typename Other_type>
void tensor<T_type>::pow(const tensor<Other_type> tensor_exponent) {
    for(int i = 0; i < size(this); i++) {
        *(this->flat_tensor_ptr + (offset + i)).pow(tensor_exponent[i]);
    }
}

template<typename T_type>
template<typename Other_type>
tensor<T_type> tensor<T_type>::dot(tensor<Other_type>& inp_tensor) {
    int n = 0;
    int max_possible_n = std::min(this->dims.size(), inp_tensor.dims.size());

    for (int k = max_possible_n; k >= 1; --k) {
        bool match = true;
        for (int i = 0; i < k; ++i) {
            if (this->dims[this->dims.size() - k + i] != inp_tensor.dims[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            n = k;
            break;
        }
    }

    auto A_tensor = this->copy();
    auto B_tensor = inp_tensor.copy();

    int rows_A = 1;
    for (size_t i = 0; i < this->dims.size() - n; ++i) rows_A *= this->dims[i];


    int cols_A = 1;
    for (size_t i = this->dims.size() - n; i < this->dims.size(); ++i) cols_A *= this->dims[i];

    int cols_B = 1;
    for (size_t i = this->dims.size() - n; i < this->dims.size(); ++i) cols_B *= this->dims[i];

    int rows_B = rows_A;

    std::vector<int> out_dims;
    for (size_t i = 0; i < this->dims.size() - n; ++i) out_dims.push_back(this->dims[i]);
    for (size_t i = n; i < inp_tensor.dims.size(); ++i) out_dims.push_back(inp_tensor.dims[i]);
    
    if (out_dims.empty()) out_dims = {1};

    std::vector<T_type> res_tensor(rows_A * cols_B, 0);

    for (int i = 0; i < rows_A; ++i) {
        for (int k = 0; k < cols_A; ++k) {
            T_type val_A = A_tensor.flat_tensor[i * cols_A + k];
            for (int j = 0; j < cols_B; ++j) {
                res_tensor[i * cols_B + j] += val_A * B_tensor.flat_tensor[k * cols_B + j];
            }
        }
    }

    return tensor<T_type>(out_dims, res_tensor);

}

template<typename T_type>
tensor<T_type> tensor<T_type>::operator[](int index) {
    index = (index % dims[0] + dims[0]) % dims[0];

    std::vector<int> out_dims {dims};

    if(size(out_dims) != 1) {
        out_dims.erase(out_dims.begin());
    }
    else {
        out_dims[0] = 1;
    }

    std::vector<int> out_strides {strides};
    if(size(out_strides) != 1) {
        out_strides.erase(out_strides.begin());
    }

    size_t out_offset = offset + index * strides[0];
    
    return tensor<T_type> (out_dims, flat_tensor, flat_tensor_ptr, out_strides, out_offset);
}

template<typename T_type>
tensor<T_type> tensor<T_type>::operator[](std::optional<int> start, std::optional<int> finish, std::optional<int> step) {
    int final_step = step.value_or(1);
    int final_start = 0;
    int final_finish = 0;
    
    if(not step.has_value() or step.value() > 0){
        if(start.has_value()) { if(start.value() < 0){final_start = (start.value() % dims[0] + dims[0]) % dims[0];} else {final_start = start.value();}} else {final_start = 0;}
        if(finish.has_value()) { if(finish.value() < 0){final_finish = (finish.value() % dims[0] + dims[0]) % dims[0];} else {final_finish = finish.value();}} else {final_finish = dims[0];}
        if(step.has_value()) { final_step = step.value();} else {final_step = 1;} 
    }
    else {
        if(start.has_value()) { if(start.value() < 0){final_start = (start.value() % dims[0] + dims[0]) % dims[0];} else {final_start = start.value();}} else {final_start = dims[0] - 1;}
        if(finish.has_value()) { if(finish.value() < 0){final_finish = (finish.value() % dims[0] + dims[0]) % dims[0];} else {final_finish = finish.value();}} else {final_finish = -1;}
        if(step.has_value()) { final_step = step.value();}
    }

    std::vector<int> out_dims {dims};
    out_dims[0] = (abs(final_finish - final_start) + abs(final_step) - 1) / abs(final_step);
    
    std::vector<int> out_strides {strides};
    out_strides[0] *= final_step;

    if(out_dims[0] == 1 and size(out_dims) != 1) {
        out_dims.erase(out_dims.begin());
    }
    
    size_t out_offset = offset + final_start * strides[0];

    return tensor<T_type> (out_dims, flat_tensor, flat_tensor_ptr, out_strides, out_offset);
}

template<typename T_type>
template<typename Other_type>
void tensor<T_type>::operator=(tensor<Other_type> inp_tensor) {
    std::copy(inp_tensor.begin(), inp_tensor.end(), flat_tensor_ptr + offset);
}

template<typename T_type>
void tensor<T_type>::operator=(const auto& scalar) {
    *(flat_tensor_ptr + offset) = scalar;
}

template<typename T_type>
tensor<T_type> tensor<T_type>::copy() {
    std::vector<T_type> contiguous_flat_tensor;
    auto collect = [&](auto& self, size_t current_offset, size_t dim_idx) -> void {
        if (dim_idx == dims.size()) {
            contiguous_flat_tensor.push_back(*(flat_tensor_ptr + current_offset));
            return;
        }
        for (int i = 0; i < dims[dim_idx]; ++i) {
            self(self, current_offset + i * strides[dim_idx], dim_idx + 1);
        }
    };

    if (!dims.empty() and count > 0) {
        collect(collect, offset, 0);
    }

    return tensor<T_type> (dims ,contiguous_flat_tensor);
}

template<typename T_type>
template<typename ... Args>
tensor<T_type>& tensor<T_type>::reshape(Args...args) {
    dims = std::vector<int> {static_cast<int>(args) ...};
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
    for(int d: dims){
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



int main(){
    std::array<float, 4> inp {1, 2, 3, 4};
    std::array<std::array<float, 5>, 4> w {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}};
    tensor<float> inp_t {inp};
    tensor<float> w_t {w};

    tensor<float> res = inp_t.dot(w_t);
    for(float i: res.flat_tensor){std::cout<<i<<std::endl;}
}