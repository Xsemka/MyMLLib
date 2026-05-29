#include <iostream>
#include <iterator>
#include <vector>
#include <array>
#include <type_traits>
using namespace std;


template<typename T_type>
class tensor{
    public:
        unsigned int count = 1;
        vector<unsigned int> strides;
        vector<unsigned int> dim;
        vector<T_type> flat_tensor;
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
            this -> dim = other_tensor.dim;
            this -> count = other_tensor.count;
            this -> strides = other_tensor.strides;
            this -> flat_tensor.reserve(other_tensor.count);
            this -> flat_tensor.assign(other_tensor.flat_tensor.begin(), other_tensor.flat_tensor.end());
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


    private:
        void calc_count_of_elements(){
            for(unsigned int d: dim){
                count *= d;
            }
            return;
        }

        template<typename Current_type>
        void extract_dim(const Current_type& current_dim){
            if constexpr (is_arithmetic_v<Current_type>) {
                return;
            }
            else{
                dim.push_back(size(current_dim));
                if(size(current_dim)>0){extract_dim(current_dim[0]);}
            }
        }
        
        template<typename Current_type>
        void extract_flat_data(const Current_type& current_tensor){
            if constexpr (is_arithmetic_v<Current_type>) {
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
            strides.resize(size(dim));
            int current_stride = 1;
            for(int i = size(dim)-1; i >= 0; i--){
                strides[i] = current_stride;
                current_stride *= dim[i];
            }
        }     
};  

int main(){
    array<array<int, 3>, 3> a {{{1,2,3}, {4,5,6}, {7, 8, 9}}};
    tensor<float> t{a};
    tensor t2 = t/2;
    for(float i: t2.flat_tensor){cout<<i<<" ";}
    
}

