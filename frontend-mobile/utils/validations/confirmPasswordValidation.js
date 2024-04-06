import * as yup from 'yup';

const confirmPasswordValidationSchema = yup.object().shape({
	password: yup
		.string()
		.min(8, ({ min }) => `Password must be at least ${min} characters`)
		.matches(/[0-9]/, 'Password requires a number')
		.matches(/[a-z]/, 'Password requires a lowercase letter')
		.matches(/[A-Z]/, 'Password requires an uppercase letter')
		.matches(/[^\w]/, 'Password requires a symbol')
		.required('Password is required'),
	confirmPassword: yup
		.string()
		.oneOf([yup.ref('password'), null], 'Passwords must match')
		.required('Please Confirm Your Password.'),
});

export default confirmPasswordValidationSchema;
