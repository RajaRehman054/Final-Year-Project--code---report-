import { useContext, useState } from 'react';
import {
	StyleSheet,
	Image,
	View,
	Text,
	TouchableOpacity,
	Keyboard,
	TouchableWithoutFeedback,
	ActivityIndicator,
} from 'react-native';
import { TextInput } from 'react-native-paper';
import { Formik } from 'formik';

import { colors } from '../../utils/theme';
import loginValidationSchema from '../../utils/validations/loginValidation';
import url from '../../utils/url.js';
import { getFCM } from '../../utils/pushNotifications';
import { AuthContext } from '../../auth/AuthContext';

const Login = ({ navigation }) => {
	const { logIn } = useContext(AuthContext);
	const [show, setShow] = useState(false);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState('');

	const handleSubmit = async values => {
		setLoading(true);
		const token = await getFCM();
		const opts1 = {
			method: 'POST',
			headers: {
				'content-type': 'application/json',
			},
			body: JSON.stringify({
				username: values.username,
				password: values.password,
			}),
		};
		const res1 = await fetch(`${url}/users/login`, opts1);
		if (!res1.ok) {
			setLoading(false);
			setError('Invalid Username or Password.');
			setInterval(() => {
				setError('');
			}, 10000);
		} else {
			const data = await res1.json();
			const opts2 = {
				method: 'PATCH',
				headers: {
					Authorization: `Bearer ${data.token}`,
					'Content-type': 'application/json',
				},
				body: JSON.stringify({
					token: token,
				}),
			};
			const res2 = await fetch(`${url}/users/fcm`, opts2);
			if (res2.ok) {
				logIn(data.token);
			} else {
				setLoading(false);
				setError('Something went Wrong');
				setInterval(() => {
					setError('');
				}, 10000);
			}
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
						<Text style={styles.text}>
							Sign In to{' '}
							<Text style={styles.textc}>Newswave</Text>
						</Text>
					</View>
					<Formik
						validationSchema={loginValidationSchema}
						initialValues={{
							username: '',
							password: '',
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
								<TouchableOpacity
									style={styles.touchforget}
									onPress={() =>
										navigation.navigate('ForgotPassword')
									}>
									<Text style={styles.textlight}>
										Forget Your Password?
									</Text>
								</TouchableOpacity>
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
										values.password === '' ||
										values.username === ''
									}>
									{loading ? (
										<ActivityIndicator
											size='small'
											color='#ffffff'
										/>
									) : (
										<Text style={styles.textsign}>
											Sign In
										</Text>
									)}
								</TouchableOpacity>
							</View>
						)}
					</Formik>
				</View>
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
				<TouchableOpacity
					style={styles.touchbottom}
					onPress={() => navigation.navigate('Register')}>
					<Text style={styles.textbottom}>
						Don't have an account?
						<Text style={styles.textc}>Sign Up</Text>
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
		marginBottom: '5%',
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
		marginTop: 10,
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
		marginTop: '10%',
		backgroundColor: colors.primary,
		borderRadius: 40,
		height: '15%',
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

export default Login;
