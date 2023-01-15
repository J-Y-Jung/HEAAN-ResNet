#include <iostream>
#include <omp.h>
#include <optional>
#include "HEaaN/heaan.hpp"
#include "HEaaNTimer.hpp"
#include "AvgpoolFC64.hpp"
#include "convtools.hpp"

namespace {
using namespace HEaaN;
using namespace std;
}

//test code
int main() {
    HEaaN::HEaaNTimer timer(false);
    // You can use other bootstrappable parameter instead of FGb.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
    const auto log_slots = getLogFullSlots(context);

    HEaaN::SecretKey sk(context);
    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate encryption key ... " << std::endl;
    keygen.genEncryptionKey();
    std::cout << "done" << std::endl << std::endl;

    /*
    You should perform makeBootstrappble function
    before generating evaluation keys and constucting HomEvaluator class.
    */
    HEaaN::makeBootstrappable(context);

    std::cout << "Generate commonly used keys (mult key, rotation keys, "
        "conjugation key) ... "
        << std::endl;
    keygen.genCommonKeys();
    std::cout << "done" << std::endl << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);

    /*
    HomEvaluator constructor pre-compute the constants for bootstrapping.
    */
    std::cout << "Generate HomEvaluator (including pre-computing constants for "
        "bootstrapping) ..."
        << std::endl;
    timer.start("* ");
    HEaaN::HomEvaluator eval(context, pack);
    timer.end();

    // ///////////// Preset ///////////////////
    // std::int rb_num;
    // rb_num = 16;
    
    

    ///////////// Message & Ctxt ///////////////////
    HEaaN::Message msg(log_slots);
    // fillRandomComplex(msg);
    std::optional<size_t> num;
    size_t length = num.has_value() ? num.value() : msg.getSize();
    size_t idx = 0;
	
    for (; idx < length; ++idx) {
        msg[idx].real(0.5);
        msg[idx].imag(0.0);
    }
    // If num is less than the size of msg,
    // all remaining slots are zero.
	
    for (; idx < msg.getSize(); ++idx) {
        msg[idx].real(0.0);
        msg[idx].imag(0.0);
    }

    printMessage(msg);
	
	HEaaN::EnDecoder ecd(context);
    HEaaN::Plaintext ptxt(context);
	ptxt = ecd.encode(msg, 1, 0);
	
    HEaaN::Ciphertext ctxt(context);
    std::cout << "Encrypt ... " << std::endl;
    enc.encrypt(ptxt, pack, ctxt); // public key encryption
    std::cout << "done" << std::endl;

    std::vector<HEaaN::Ciphertext> ctxt_bundle;
	std::vector<vector<HEaaN::Plaintext>> ptxt_bundle;
    for (int i = 0; i < 64; ++i) {
    	ctxt_bundle.push_back(ctxt);
    }
	vector<HEaaN::Plaintext> ptxt_bundle_tmp;
	for (int i = 0; i < 64; ++i) {
		ptxt_bundle_tmp.push_back(ptxt);
	}
	for (int i = 0; i < 64; ++i) {
		ptxt_bundle.push_back(ptxt_bundle_tmp);
    }
	HEaaN::Ciphertext tmp_ctxt(context);
	std::vector<HEaaN::Ciphertext> ctxt_out(10, tmp_ctxt);
	timer.start(" FC64 ");
	ctxt_out=FC64(context, pack, eval, 
                    ctxt_bundle, ptxt_bundle, ptxt_bundle_tmp);
	timer.end();
    std::cout << "FC64 is over" << "\n";
	
	/////////////// Decryption ////////////////
    HEaaN::Message dmsg;
	std::cout << "level:" << ctxt_out[0].getLevel();
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt_out[0], sk, dmsg);
    std::cout << "done" << std::endl;
    printMessage(dmsg);
	
    return 0;
}