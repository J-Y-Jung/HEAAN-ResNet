//demo IO file.



//Setting
HEaaN::Ciphertext MPPacking(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, int imgsize, 
std::vector<HEaaN::Ciphertext> ctxt_bundle);
HEaaN::Ciphertext RotSumToIdx(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
                              u64 rot_interval, u64 log_rot_num, u64 idx, HEaaN::Ciphertext ctxt);



//conv
HEaaN::Ciphertext Conv(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, int imgsize, int gap, int stride, HEaaN::Ciphertext ctxt, 
std::vector<HEaaN::Message> kernel_bundle);

void printMessage(const Message &msg, bool is_complex = true,
                  size_t start_num = 64, size_t end_num = 64);

void Weight2Msg(Message &msg, vector<vector<double>> weights,
                    u64 gap_in, u64 stride, 
                    u64 weight_row_idx, u64 weight_col_idx);

void Weight2Msg1stConv(Message &msg, vector<vector<double>> weights, 
                    u64 weight_row_idx, u64 weight_col_idx);
                    


//BN
void BN(const double mu , const double std , const double gamma, const double beta ,
        HEaaN::Ciphertext ctxt , HEaaN::HomEvaluator eval, HEaaN::Ciphertext context);

//ReLU
void print_polynomial(const std::vector<double> &polynomial, const size_t degree);
void SetUp(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext& ctxt,
    std::vector<HEaaN::Ciphertext>& BS_basis,
    std::vector<HEaaN::Ciphertext>& GS_basis,
    const int k, const int l);
void BabyStep(HEaaN::Context context, HEaaN::HomEvaluator eval,
                           const std::vector<HEaaN::Ciphertext> &basis,
                           const std::vector<double> &polynomial,
                           HEaaN::Ciphertext &ctxt_result,
                           const int length);
std::vector<double> vectorSlice(const std::vector<double> &input, int a, int b);
void GiantStep(HEaaN::Context context, HEaaN::HomEvaluator eval, 
                            const std::vector<HEaaN::Ciphertext> &BS_basis, 
                            const std::vector<HEaaN::Ciphertext> &GS_basis, 
                            const std::vector<double> &polynomial,
                            HEaaN::Ciphertext &ctxt_result,
                            int k, int l);
void evalPolynomial(HEaaN::Context context, HEaaN::HomEvaluator eval, 
                                HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_poly,
                                const std::vector<double> &polynomial);
void ApproxReLU(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_relu);

//AVGPOOL

//FClayer
