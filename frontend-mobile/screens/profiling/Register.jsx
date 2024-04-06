import { useState } from 'react';
import {
	StyleSheet,
	Image,
	View,
	Text,
	TouchableOpacity,
	TouchableWithoutFeedback,
	Keyboard,
	ActivityIndicator,
} from 'react-native';
import { TextInput } from 'react-native-paper';
import Toast from 'react-native-toast-message';
import { Formik } from 'formik';

import { colors } from '../../utils/theme';
import registerValidationSchema from '../../utils/validations/registerValidation';

const Register = ({ navigation }) => {
	const [show, setShow] = useState(false);
	const [error, setError] = useState('');
	const [loading, setLoading] = useState(false);

	const handleSubmit = async values => {
		setLoading(true);
		const opts1 = {
			method: 'POST',
			headers: {
				'content-type': 'application/json',
			},
			body: JSON.stringify({
				username: values.username,
				password: values.password,
				email: values.email,
			}),
		};
		const res = await fetch(`${url}/users/register`, opts1);
		const data = await res.json();
		if (res.status !== 201) {
			setLoading(false);
			setError(data.message);
			setInterval(() => {
				setError('');
			}, 10000);
		} else {
			Toast.show({
				type: 'success',
				text1: 'User Created Successfully ✅',
			});
			navigation.navigate('Login');
		}
	};
	return (
		<TouchableWithoutFeedback onPress={Keyboard.dismiss}>
			<View style={styles.scrollview}>
				<View style={styles.curlyview}></View>
				<View style={styles.mainview}>
					<View style={styles.imgview}>
						<Image
							source={require('../../assets/orange-logo.png')}
							style={styles.img}></Image>
						<Text style={styles.text}>Create New Account</Text>
						{error !== '' ? (
							<Text
								style={[
									styles.textbottom,
									{
										color: 'red',
										textAlign: 'center',
										fontWeight: 'bold',
									},
								]}>
								{error}
							</Text>
						) : null}
					</View>
					<Formik
						validationSchema={registerValidationSchema}
						initialValues={{
							username: '',
							password: '',
							confirmPassword: '',
							email: '',
						}}
						onSubmit={values => handleSubmit(values)}>
						{({
							handleChange,
							handleBlur,
							handleSubmit,
							errors,
							isValid,
							touched,
							values,
						}) => (
							<View>
								<TextInput
									outlineColor={colors.primary}
									theme={{
										colors: {
											text: 'black',
											primary: colors.orange,
										},
										roundness: 50,
									}}
									style={styles.textinput}
									mode='outlined'
									value={values.username}
									onChangeText={handleChange('username')}
									onBlur={handleBlur('username')}
									placeholder='Username'
								/>
								{errors.username && touched.username && (
									<Text style={styles.errorText}>
										{errors.username}
									</Text>
								)}
								<TextInput
									outlineColor={colors.primary}
									theme={{
										colors: {
											text: 'black',
											primary: colors.orange,
										},
										roundness: 50,
									}}
									style={styles.textinput}
									mode='outlined'
									value={values.email}
									onChangeText={handleChange('email')}
									onBlur={handleBlur('email')}
									placeholder='Email'
								/>
								{errors.email && touched.email && (
									<Text style={styles.errorText}>
										{errors.email}
									</Text>
								)}
								<TextInput
									style={styles.textinput}
									mode='outlined'
									value={values.password}
									onChangeText={handleChange('password')}
									onBlur={handleBlur('password')}
									placeholder='Password'
									outlineColor={colors.primary}
									theme={{
										colors: {
											text: 'black',
											primary: colors.orange,
										},
										roundness: 50,
									}}
									secureTextEntry={show ? false : true}
									right={
										<TextInput.Icon
											icon='eye'
											onPress={() =>
												setShow(prevShow => !prevShow)
											}
										/>
									}
								/>
								{errors.password && touched.password && (
									<Text style={styles.errorText}>
										{errors.password}
									</Text>
								)}
								<TextInput
									style={styles.textinput}
									mode='outlined'
									value={values.confirmPassword}
									onChangeText={handleChange(
										'confirmPassword'
									)}
									onBlur={handleBlur('confirmPassword')}
									placeholder='Confirm Password'
									outlineColor={colors.primary}
									theme={{
										colors: {
											text: 'black',
											primary: colors.orange,
										},
										roundness: 50,
									}}
									secureTextEntry={show ? false : true}
									right={
										<TextInput.Icon
											icon='eye'
											onPress={() =>
												setShow(prevShow => !prevShow)
											}
										/>
									}
								/>
								{errors.confirmPassword &&
									touched.confirmPassword && (
										<Text style={styles.errorText}>
											{errors.confirmPassword}
										</Text>
									)}
								<TouchableOpacity
									style={[
										styles.touchsign,
										{
											backgroundColor: !isValid
												? '#FFA559'
												: colors.primary,
										},
									]}
									onPress={handleSubmit}
									disabled={
										!isValid ||
										values.email === '' ||
										values.password === '' ||
										values.confirmPassword === '' ||
										values.username === ''
									}>
									{loading ? (
										<ActivityIndicator
											size='small'
											color='#ffffff'
										/>
									) : (
										<Text style={styles.textsign}>
											Sign Up
										</Text>
									)}
								</TouchableOpacity>
							</View>
						)}
					</Formik>
				</View>
				<TouchableOpacity
					style={styles.touchbottom}
					onPress={() => navigation.navigate('Login')}>
					<Text style={styles.textbottom}>
						Already have an account?
						<Text style={styles.textc}>Sign In</Text>
					</Text>
				</TouchableOpacity>
			</View>
		</TouchableWithoutFeedback>
	);
};

const styles = StyleSheet.create({
	scrollview: {
		flex: 1,
		backgroundColor: 'white',
	},
	curlyview: {
		height: '10%',
		backgroundColor: colors.primary,
		marginBottom: '5%',
		borderBottomStartRadius: 80,
	},
	mainview: {
		padding: '3%',
	},
	imgview: {
		alignItems: 'center',
	},
	img: {
		height: 100,
		resizeMode: 'cover',
		marginBottom: '3%',
	},
	text: {
		fontFamily: 'Poppins',
		color: 'black',
		fontWeight: '900',
		fontSize: 20,
		marginBottom: '10%',
	},
	textc: {
		color: colors.primary,
	},
	textinput: {
		backgroundColor: colors.lightOrange,
		marginTop: 5,
		fontSize: 13,
	},
	textlight: {
		fontFamily: 'Poppins',
		fontSize: 14.5,
		color: 'grey',
	},
	touchforget: {
		alignSelf: 'flex-end',
		paddingTop: 10,
	},
	touchsign: {
		marginTop: '5%',
		backgroundColor: colors.primary,
		borderRadius: 40,
		height: '12%',
		justifyContent: 'center',
		alignItems: 'center',
	},
	textsign: {
		color: 'mintcream',
		fontFamily: 'Poppins',
		fontWeight: 'bold',
		fontSize: 16,
	},
	textbottom: {
		color: 'black',
	},
	touchbottom: {
		justifyContent: 'center',
		alignItems: 'center',
		flex: 1,
	},
	errorText: {
		fontSize: 10,
		color: 'red',
		alignSelf: 'flex-end',
	},
});

export default Register;
