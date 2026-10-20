import unittest
from profile_instrument import instrument, instrument_nightly_device


class InstrumentTests(unittest.TestCase):
    def test_end_precedes_park_and_verification(self):
        source='''int main() {
mu_schedule(stage, nullptr, 4);
mu_barrier(0, MU_NUM_CORES);
mu_fence();
if (mu_hart_id() != 0) { for (;;) {} }
verify_output(); mu_tohost(0);
}'''
        result=instrument(source)
        self.assertLess(result.index('NIGHTLY_PROFILE_BEGIN'),result.index('mu_schedule(stage'))
        self.assertLess(result.index('NIGHTLY_PROFILE_END'),result.index('verify_output'))
        self.assertEqual(result.count('mu_schedule(stage'),1)

    def test_model_input_initializer_stays_outside_work(self):
        source='''int main() {
mu_schedule(init_inputs, nullptr, 4); mu_barrier(0, MU_NUM_CORES); mu_fence();
mu_schedule(stage, nullptr, 4); mu_barrier(0, MU_NUM_CORES); mu_fence();
if(vx_core_id()!=0)for(;;){}
mu_tohost(0);
}'''
        result=instrument(source)
        self.assertLess(result.index('mu_schedule(init_inputs'),result.index('NIGHTLY_PROFILE_BEGIN'))
        self.assertLess(result.index('NIGHTLY_PROFILE_BEGIN'),result.index('mu_schedule(stage'))

    def test_missing_boundary_is_an_error(self):
        with self.assertRaises(ValueError):instrument('int main(){ mu_schedule(stage,0,4); }')

    def test_changed_nightly_template_is_an_error(self):
        with self.assertRaises(ValueError):instrument_nightly_device('unknown device helper')


if __name__=='__main__':unittest.main()
