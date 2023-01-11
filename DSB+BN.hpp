namespace {
using namespace HEaaN;
using namespace std;
}

vector<vector<Ciphertext>> DSB(Context context, KeyPack pack,
HomEvaluator eval, int DSB_count, vector<vector<Ciphertext>> ctxt_bundle, 
// 첫번째 index는 서로 다른 이미지 index. 기본 처음에는 16. 첫번째 DSB에서는 16개로 받음. 두번째는 4개. 두번째 : ch
vector<vector<vector<Plaintext>>> kernel_bundle, 
vector<vector<vector<Plaintext>>> kernel_bundle2, 
vector<vector<vector<Plaintext>>> kernel_residual_bundle,
vector<vector<Plaintext>> BN_sum1,
vector<vector<Plaintext>> BN_sum2) {
    ///////////////////////// SetUp ////////////////////////////////
    cout << "DSB start" << "\n";
    // int num_ctxt;
    // num_ctxt = ctxt_bundle.size();

    // int num_kernel_bundle1;
    // num_kernel_bundle1 = kernel_bundle.size();

    // int num_kernel_bundle2;
    // num_kernel_bundle2 = kernel_bundle2.size();


    ///////////////////////// Main flow /////////////////////////////////////////
    cout << "First Conv-(main flow) ..." << endl;
    cout << "level of ctxt is " << ctxt_bundle[0][0].getLevel() << "\n";
    vector<vector<Ciphertext>> ctxt_conv_out_bundle;
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        vector<Ciphertext> ctxt_conv_out_cache;
        ctxt_conv_out_cache = Conv(context, pack, eval, 32, 1, 2, 16, 32, ctxt_bundle[i], kernel_bundle);
        ctxt_conv_out_bundle.push_back(ctxt_conv_out_cache);
    }
    cout << "level of ctxt is " << ctxt_conv_out_bundle[0][0].getLevel() << "\n";

    // vector<vector<vector<Plaintext>>>().swap(kernel_bundle);
    cout << "DONE!" << "\n";
    /* 여기서 나온 ctxt_conv_out_bundle은 첫번째는 0이상 16미만의 서로다른 img 개수 인덱스,
    두번째는 0이상 32미만의 channel index
    */
    // MPP input bundle making
    cout << "MPP-(main flow, First Conv) ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_MPP_in;
    for (int i = 0; i < 4; ++i) {
        vector<vector<Ciphertext>> ctxt_MPP_in_allch_bundle;
        for (int ch = 0; ch < 32; ++ch) {
            vector<Ciphertext> ctxt_MPP_in_cache;
            ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+0][ch]);
            ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+1][ch]);
            ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+2][ch]);
            ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+3][ch]);
            ctxt_MPP_in_allch_bundle.push_back(ctxt_MPP_in_cache);
        }
        ctxt_MPP_in.push_back(ctxt_MPP_in_allch_bundle);
    }
    // vector<vector<Ciphertext>>().swap(ctxt_conv_out_bundle);

    // ctxt_MPP_in 첫번째 : 4개씩 묶음 index 0이상 4미만, 두번째 : ch, 세번째 : ctxt 4개에 대한 index
    // MPP
    vector<vector<Ciphertext>> ctxt_MPP_out_bundle;
    for (int i = 0; i < 4; ++i) {
        vector<Ciphertext> ctxt_MPP_out;
        for (int ch = 0; ch < 32; ++ch) {
            Ciphertext ctxt_MPP_out_cache(context);
            ctxt_MPP_out_cache = MPPacking(context, pack, eval, 32, ctxt_MPP_in[i][ch]);
            ctxt_MPP_out.push_back(ctxt_MPP_out_cache);
        }
        ctxt_MPP_out_bundle.push_back(ctxt_MPP_out);
    }
    // vector<vector<vector<Ciphertext>>>().swap(ctxt_MPP_in);
    cout << "DONE!" << "\n";

    
    // ctxt_MPP_out_bundle 첫번째 : 서로 다른 img, 두번째 : ch.

    // /////////////// Decryption ////////////////
    // Message dmsg0;
    // cout << "Decrypt ... ";
    // dec.decrypt(ctxt_MPP_out, sk, dmsg0);
    // cout << "done" << endl;
    // printMessage(dmsg0);
    // ////////////////////////////////////////

    // AppReLU
    cout << "AppReLU-(main flow) ..." << endl;
    vector<vector<Ciphertext>> ctxt_relu_out_bundle;
    for (int i = 0; i < 4; ++i) {
        vector<Ciphertext> ctxt_relu_out_allch_bundle;
        for (int ch = 0; ch < 32; ++ch) {
            cout << "(i = " << i << ", " << "ch = " << ch << ")" << "\n";
            Ciphertext ctxt_relu_out(context);
            ApproxReLU(context, eval, ctxt_MPP_out_bundle[i][ch], ctxt_relu_out);
            ctxt_relu_out_allch_bundle.push_back(ctxt_relu_out);
        }
        ctxt_relu_out_bundle.push_back(ctxt_relu_out_allch_bundle);
    }
    // vector<vector<Ciphertext>>().swap(ctxt_MPP_out_bundle);
    cout << "DONE!" << "\n";

    // Second convolution
    cout << "Second Conv-(main flow) ..." << endl;
    vector<vector<Ciphertext>> ctxt_conv_out2_bundle;
    for (int i = 0; i < 4; ++i) {
        vector<Ciphertext> ctxt_conv_out2_allch_bundle;
        ctxt_conv_out2_allch_bundle = Conv(context, pack, eval, 32, 1, 1, 32, 32, ctxt_relu_out_bundle[i], kernel_bundle2);
        ctxt_conv_out2_bundle.push_back(ctxt_conv_out2_allch_bundle);
    }
    cout << "level of ctxt is " << ctxt_conv_out2_bundle[0][0].getLevel() << "\n";
    // vector<vector<vector<Plaintext>>>().swap(kernel_bundle2);
    // vector<vector<Ciphertext>>().swap(ctxt_relu_out_bundle);
    cout << "DONE!" << "\n";

    ///////////////////// Residual flow ////////////////////////////
    // Convolution
    cout << "Residual Conv-(residual flow) ..." << endl;
    vector<vector<Ciphertext>> ctxt_residual_out_bundle;
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        vector<Ciphertext> ctxt_residual_out_cache;
        ctxt_residual_out_cache = Conv(context, pack, eval, 32, 1, 2, 16, 32, ctxt_bundle[i], kernel_residual_bundle);
        //에러나면 push_back 해야 할 수도 있음
        ctxt_residual_out_bundle.push_back(ctxt_residual_out_cache);
    }
    // vector<vector<vector<Plaintext>>>().swap(kernel_residual_bundle);
    // vector<vector<Ciphertext>>().swap(ctxt_bundle);
    cout << "DONE!" << "\n";
    // MPP input bundle making
    cout << "MPP-(residual flow) ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_MPP_in2;
    for (int i = 0; i < 4; ++i) {
        vector<vector<Ciphertext>> ctxt_MPP_in_allch_bundle2;
        for (int ch = 0; ch < 32; ++ch) {
            vector<Ciphertext> ctxt_MPP_in_cache2;
            ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+0][ch]);
            ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+1][ch]);
            ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+2][ch]);
            ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+3][ch]);
            ctxt_MPP_in_allch_bundle2.push_back(ctxt_MPP_in_cache2);
        }
        ctxt_MPP_in2.push_back(ctxt_MPP_in_allch_bundle2);
    }
    // vector<vector<Ciphertext>>().swap(ctxt_residual_out_bundle);
    // MPP
    vector<vector<Ciphertext>> ctxt_MPP_out_bundle2;
    for (int i = 0; i < 4; ++i) {
        vector<Ciphertext> ctxt_MPP_out2;
        for (int ch = 0; ch < 32; ++ch) {
            Ciphertext ctxt_MPP_out_cache2(context);
            ctxt_MPP_out_cache2 = MPPacking(context, pack, eval, 32, ctxt_MPP_in2[i][ch]);
            ctxt_MPP_out2.push_back(ctxt_MPP_out_cache2);
        }
        ctxt_MPP_out_bundle2.push_back(ctxt_MPP_out2);
    }
    cout << "level of ctxt is " << ctxt_MPP_out_bundle2[0][0].getLevel() << "\n";
    // vector<vector<vector<Ciphertext>>>().swap(ctxt_MPP_in2);
    cout << "DONE!" << "\n";


    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "Main flow + Residual flow ..." << endl;
    vector<vector<Ciphertext>> ctxt_residual_added;
    for (int i = 0; i < 4; ++i) {
        vector<Ciphertext> ctxt_residual_added_allch_bundle;
        for (int ch = 0; ch < 32; ++ch) {
            Ciphertext ctxt_residual_added_cache(context);
            eval.add(ctxt_conv_out2_bundle[i][ch], ctxt_MPP_out_bundle2[i][ch], ctxt_residual_added_cache);
            ctxt_residual_added_allch_bundle.push_back(ctxt_residual_added_cache);
        }
        ctxt_residual_added.push_back(ctxt_residual_added_allch_bundle);
    }
    // vector<vector<Ciphertext>>().swap(ctxt_conv_out2_bundle);
    // vector<vector<Ciphertext>>().swap(ctxt_MPP_out_bundle2);
    cout << "DONE!" << "\n";

    // Last AppReLU
    cout << "Last AppReLU ..." << endl;
    vector<vector<Ciphertext>> ctxt_DSB_out;
    for (int i = 0; i < 4; ++i) {
        vector<Ciphertext> ctxt_DSB_out_allch_bundle;
        for (int ch = 0; ch < 32; ++ch) {
            cout << "(i = " << i << ", " << "ch = " << ch << ")" << "\n";
            Ciphertext ctxt_DSB_out_cache(context);
            ApproxReLU(context, eval, ctxt_residual_added[i][ch], ctxt_DSB_out_cache);
            ctxt_DSB_out_allch_bundle.push_back(ctxt_DSB_out_cache);
        }
        ctxt_DSB_out.push_back(ctxt_DSB_out_allch_bundle);
    }
    // vector<vector<Ciphertext>>().swap(ctxt_residual_added);
    cout << "DONE!" << "\n";


    return ctxt_DSB_out;
}
