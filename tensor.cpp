#include <algorithm>
#include <cstddef> 
#include <iostream>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>
#include <array>
#include <type_traits>


template<typename T_type>
class tensor{
    public:
        unsigned int count = 1;
        std::vector<unsigned int> strides;
        std::vector<unsigned int> dims;
        std::vector<T_type> flat_tensor;
        template<typename InpType>
        tensor(const InpType& inp_tensor){
            extract_dim(inp_tensor);
            calc_count_of_elements();
            flat_tensor.reserve(count);
            extract_flat_data(inp_tensor);     
            calc_strides();
        }

        template<typename Other_type>
        tensor(const tensor<Other_type>& other_tensor) {
            this -> dims = other_tensor.dim;
            this -> count = other_tensor.count;
            this -> strides = other_tensor.strides;
            this -> flat_tensor.reserve(other_tensor.count);
            this -> flat_tensor.assign(other_tensor.flat_tensor.begin(), other_tensor.flat_tensor.end());
        }

        tensor(std::vector<unsigned int> given_dims, std::vector<T_type> given_flat_tensor): dims(std::move(given_dims)), flat_tensor(std::move(given_flat_tensor)) {
            calc_count_of_elements();
            calc_strides();
        }
        
        tensor& operator+=(const auto& scalar) {
            for(int i = 0; i<count; i++){
                flat_tensor[i] += scalar;
            }
            return *this;
        }

        template<typename Other_type>
        tensor& operator+=(const tensor<Other_type>& tensor) {
            for(int i = 0; i<count; i++){
                flat_tensor[i] += tensor.flat_tensor[i];
            }
            return *this;
        }

        tensor& operator+(const auto& scalar) {
            return *this += scalar;
        }

        template<typename Other_type>
        tensor& operator+(const tensor<Other_type>& tensor) {
            return *this += tensor;
        }

        tensor& operator-=(const auto& scalar) {
            for(int i = 0; i<count; i++){
                flat_tensor[i] -= scalar;
            }
            return *this;
        }

        template<typename Other_type>
        tensor& operator-=(const tensor<Other_type>& tensor) {
            for(int i = 0; i<count; i++){
                flat_tensor[i] -= tensor.flat_tensor[i];
            }
            return *this;
        }

        tensor& operator-(const auto& scalar) {
            return *this -= scalar;
        }

        template<typename Other_type>
        tensor& operator-(const tensor<Other_type>& tensor) {
            return *this -= tensor;
        }

        tensor& operator*=(const auto& scalar) {
            for(int i = 0; i<count; i++){
                flat_tensor[i] *= scalar;
            }
            return *this;
        }

        template<typename Other_type>
        tensor& operator*=(const tensor<Other_type>& tensor) {
            for(int i = 0; i<count; i++){
                flat_tensor[i] *= tensor.flat_tensor[i];
            }
            return *this;
        }

        tensor& operator*(const auto& scalar) {
            return *this *= scalar;
        }

        template<typename Other_type>
        tensor& operator*(const tensor<Other_type>& tensor) {
            return *this *= tensor;
        }

        tensor& operator/=(const auto& scalar) {
            for(int i = 0; i<count; i++){
                flat_tensor[i] /= scalar;
            }
            return *this;
        }

        template<typename Other_type>
        tensor& operator/=(const tensor<Other_type>& tensor) {
            for(int i = 0; i<count; i++){
                flat_tensor[i] /= tensor.flat_tensor[i];
            }
            return *this;
        }

        tensor& operator/(const auto& scalar) {
            return *this /= scalar;
        }

        template<typename Other_type>
        tensor& operator/(const tensor<Other_type>& tensor) {
            return *this /= tensor;
        }

        tensor operator[](int start = 0, std::optional<int> finish = std::nullopt, int step = 1) {
            finish = finish.value_or(dims[0]);

            std::vector<unsigned int> out_dims {dims};
            out_dims[0] = (finish.value() - start + step - 1) / step;

            std::vector<T_type> out_flat_tensor;

            for(int i = start; i < finish.value(); i += step){
                for(int j = i*strides[0]; j < i*strides[0] + strides[0]; j++) {
                    out_flat_tensor.push_back(flat_tensor[j]);
                }
            }
            return tensor<T_type> (std::move(out_dims), std::move(out_flat_tensor));
            
        }


    private:
        void calc_count_of_elements(){
            for(unsigned int d: dims){
                count *= d;
            }
            return;
        }

        template<typename Current_type>
        void extract_dim(const Current_type& current_dim){
            if constexpr (std::is_arithmetic_v<Current_type>) {
                return;
            }
            else{
                dims.push_back(size(current_dim));
                if(size(current_dim)>0){extract_dim(current_dim[0]);}
            }
        }
        
        template<typename Current_type>
        void extract_flat_data(const Current_type& current_tensor){
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

        void calc_strides(){
            strides.resize(size(dims));
            int current_stride = 1;
            for(int i = size(dims)-1; i >= 0; i--){
                strides[i] = current_stride;
                current_stride *= dims[i];
            }
        }     
};  

int main(){
    std::array<std::array<int, 3>, 2> a {{{1,2,3}, {4,5,6},}};
    tensor<float> t{a};
    for(float i: t.flat_tensor){std::cout<<i<<" ";}
    std::cout<<std::endl;
    for(float i: t.dims){std::cout<<i<<" ";}
    std::cout<<std::endl;
    for(float i: t.strides){std::cout<<i<<" ";}
    std::cout<<std::endl;


    tensor<float> t2 {t[0, 2]};
    for(float i: t2.flat_tensor){std::cout<<i<<" ";}
    std::cout<<std::endl;
    for(float i: t2.dims){std::cout<<i<<" ";}
    std::cout<<std::endl;
    for(float i: t2.strides){std::cout<<i<<" ";}
    std::cout<<std::endl;
}

