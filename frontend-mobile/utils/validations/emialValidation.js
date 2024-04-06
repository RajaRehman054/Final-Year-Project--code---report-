import * as yup from 'yup';

const emailValidationSchema = yup.object().shape({
	email: yup
		.string()
		.email('Enter a valid email.')
		.required('Email is required'),
});

export default emailValidationSchema;
