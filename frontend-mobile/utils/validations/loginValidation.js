import * as yup from 'yup';

const loginValidationSchema = yup.object().shape({
	username: yup
		.string()
		.required('Username is Required')
		.min(4, ({ min }) => `Username has to be at least ${min} characters`),
	password: yup
		.string()
		.min(8, ({ min }) => `Password must be at least ${min} characters`)
		.required('Password is required'),
});

export default loginValidationSchema;
