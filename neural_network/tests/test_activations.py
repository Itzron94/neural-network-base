# test_activation_functions.py

import unittest
import numpy as np
from neural_network.core.activations import (
    ActivationFunctionType,
    ActivationFunctionFactory,
    StepActivation,
    StepBipolarActivation,
    SigmoidActivation,
    ReLUActivation
)


class TestActivationFunctions(unittest.TestCase):

    def test_factory_create_by_string(self):
        """Testea que el factory crea instancias correctas por string."""
        # Probar tipos válidos por string
        self.assertIsInstance(ActivationFunctionFactory.create("STEP"), StepActivation)
        self.assertIsInstance(ActivationFunctionFactory.create("STEP_BIPOLAR"), StepBipolarActivation)
        self.assertIsInstance(ActivationFunctionFactory.create("SIGMOID"), SigmoidActivation)
        self.assertIsInstance(ActivationFunctionFactory.create("RELU"), ReLUActivation)

        # Probar tipo no válido
        with self.assertRaises(ValueError):
            ActivationFunctionFactory.create("INVALID_TYPE")
    
    def test_factory_create_methods(self):
        """Testea los métodos del factory."""
        # Probar create por string
        self.assertIsInstance(ActivationFunctionFactory.create("STEP"), StepActivation)
        self.assertIsInstance(ActivationFunctionFactory.create("step_bipolar"), StepBipolarActivation)  # Test case insensitive
        
        # Probar create_from_enum
        self.assertIsInstance(ActivationFunctionFactory.create_from_enum(ActivationFunctionType.SIGMOID), SigmoidActivation)
        self.assertIsInstance(ActivationFunctionFactory.create_from_enum(ActivationFunctionType.RELU), ReLUActivation)
        
        # Probar tipos disponibles
        available_strings = ActivationFunctionFactory.get_available_types()
        self.assertIn("STEP", available_strings)
        self.assertIn("STEP_BIPOLAR", available_strings)
        self.assertIn("SIGMOID", available_strings)
        self.assertIn("RELU", available_strings)
        
        available_enums = ActivationFunctionFactory.get_available_enums()
        self.assertIn(ActivationFunctionType.STEP, available_enums)
        self.assertIn(ActivationFunctionType.STEP_BIPOLAR, available_enums)

    def test_step_activation(self):
        """Prueba la función de activación escalón (StepActivation)."""
        act_func = StepActivation()

        # Valores escalares
        self.assertEqual(act_func.activate(-1), 0.0)
        self.assertEqual(act_func.activate(0), 1.0)
        self.assertEqual(act_func.activate(1), 1.0)

        # Arrays de entrada
        x = np.array([-2, -1, 0, 1, 2])
        expected_output = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(act_func.activate(x), expected_output)

        # Derivada (debe ser cero en todos los puntos)
        expected_derivative = np.zeros_like(x)
        np.testing.assert_array_equal(act_func.derivative(x), expected_derivative)

    def test_step_bipolar_activation(self):
        """Prueba la función de activación escalón bipolar (StepBipolarActivation) - TP1."""
        act_func = StepBipolarActivation()

        # Valores escalares
        self.assertEqual(act_func.activate(-1), -1.0)
        self.assertEqual(act_func.activate(0), 1.0)
        self.assertEqual(act_func.activate(1), 1.0)

        # Arrays de entrada - casos específicos del TP1
        x = np.array([-2, -1, 0, 1, 2])
        expected_output = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(act_func.activate(x), expected_output)

        # Test con datos exactos del TP1
        tp1_inputs = np.array([-1.0, 0.0, 1.0])
        tp1_expected = np.array([-1.0, 1.0, 1.0])
        np.testing.assert_array_equal(act_func.activate(tp1_inputs), tp1_expected)

        # Derivada (debe ser cero en todos los puntos)
        expected_derivative = np.zeros_like(x)
        np.testing.assert_array_equal(act_func.derivative(x), expected_derivative)

        # Verificar propiedades
        self.assertEqual(act_func.name, "Step Bipolar")
        self.assertEqual(act_func.output_range, (-1.0, 1.0))

    def test_sigmoid_activation(self):
        """Prueba la función de activación sigmoide (SigmoidActivation)."""
        act_func = SigmoidActivation()

        # Valores escalares
        self.assertAlmostEqual(act_func.activate(0.0), 0.5, places=7)
        self.assertAlmostEqual(act_func.activate(-1.0), 1 / (1 + np.exp(1)), places=7)
        self.assertAlmostEqual(act_func.activate(1.0), 1 / (1 + np.exp(-1)), places=7)

        # Arrays de entrada
        x = np.array([-1000, -1, 0, 1, 1000])
        # Utilizar la versión actualizada de activate que maneja clipping
        expected_output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        np.testing.assert_allclose(act_func.activate(x), expected_output, rtol=1e-5, atol=1e-8)

        # Derivada
        sig = act_func.activate(x)
        expected_derivative = sig * (1 - sig)
        np.testing.assert_allclose(act_func.derivative(x), expected_derivative, rtol=1e-5, atol=1e-8)

    def test_relu_activation(self):
        """Prueba la función de activación ReLU (ReLUActivation)."""
        act_func = ReLUActivation()

        # Valores escalares
        self.assertEqual(act_func.activate(-1), 0.0)
        self.assertEqual(act_func.activate(0), 0.0)
        self.assertEqual(act_func.activate(1), 1.0)

        # Arrays de entrada
        x = np.array([-2, -1, 0, 1, 2])
        expected_output = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(act_func.activate(x), expected_output)

        # Derivada
        expected_derivative = np.where(x > 0, 1.0, 0.0)
        np.testing.assert_array_equal(act_func.derivative(x), expected_derivative)

    def test_activation_function_interface(self):
        """Verifica que todas las funciones de activación implementan la interfaz correctamente."""
        # Test con strings
        for act_type_str in ActivationFunctionFactory.get_available_types():
            act_func = ActivationFunctionFactory.create(act_type_str)
            self.assertTrue(hasattr(act_func, 'activate'))
            self.assertTrue(hasattr(act_func, 'derivative'))
            self.assertTrue(hasattr(act_func, 'name'))
            self.assertTrue(hasattr(act_func, 'output_range'))
            
        # Test con enums
        for act_type_enum in ActivationFunctionFactory.get_available_enums():
            act_func = ActivationFunctionFactory.create_from_enum(act_type_enum)
            self.assertTrue(hasattr(act_func, 'activate'))
            self.assertTrue(hasattr(act_func, 'derivative'))
            self.assertTrue(hasattr(act_func, 'name'))
            self.assertTrue(hasattr(act_func, 'output_range'))

    def test_activation_functions_with_extreme_values(self):
        """Prueba las funciones de activación con valores extremos."""
        x = np.array([-1e10, -1e5, 0, 1e5, 1e10])

        # StepActivation
        act_func = StepActivation()
        expected_output = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(act_func.activate(x), expected_output)

        # SigmoidActivation
        act_func = SigmoidActivation()
        # Usar la versión actualizada de activate que maneja clipping
        expected_output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        np.testing.assert_allclose(act_func.activate(x), expected_output, atol=1e-7)

        # Derivada
        sig = act_func.activate(x)
        expected_derivative = sig * (1 - sig)
        np.testing.assert_allclose(act_func.derivative(x), expected_derivative, atol=1e-7)

        # ReLUActivation
        act_func = ReLUActivation()
        expected_output = np.array([0.0, 0.0, 0.0, 1e5, 1e10])
        np.testing.assert_array_equal(act_func.activate(x), expected_output)

    def test_derivative_numerical_approximation(self):
        """Comprueba que la derivada analítica coincide con la numérica."""
        delta = 1e-5
        x = np.linspace(-1, 1, 10)

        # SigmoidActivation
        act_func = SigmoidActivation()
        numerical_derivative = (act_func.activate(x + delta) - act_func.activate(x - delta)) / (2 * delta)
        analytical_derivative = act_func.derivative(x)
        np.testing.assert_allclose(numerical_derivative, analytical_derivative, rtol=1e-4, atol=1e-6)

        # ReLUActivation
        act_func = ReLUActivation()
        numerical_derivative = (act_func.activate(x + delta) - act_func.activate(x - delta)) / (2 * delta)
        analytical_derivative = act_func.derivative(x)
        # Ignorar el punto donde x=0 debido a la discontinuidad
        mask = x != 0
        np.testing.assert_allclose(numerical_derivative[mask], analytical_derivative[mask], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
