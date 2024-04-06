import { useState } from 'react';
import {
	View,
	StyleSheet,
	Text,
	TouchableOpacity,
	TouchableWithoutFeedback,
	Keyboard,
	ActivityIndicator,
} from 'react-native';
import Ant from 'react-native-vector-icons/AntDesign';
import { Formik } from 'formik';
import { TextInput } from 'react-native-paper';
import Toast from 'react-native-toast-message';

import emailValidationSchema from '../../utils/validations/emialValidation';
import { colors } from '../../utils/theme.js';

const ForgotPassword = ({ navigation }) => {
	const [error, setError] = useState('');
	const [loading, setLoading] = useState(false);

	const main = async values => {
		setLoading(true);
		const res = await fetch(`${url}/users/otp/${values.email}`);
		if (res.ok) {
			Toast.show({
				type: 'success',
				text1: 'Kindly check your email. ✅',
			});
			navigation.navigate('EmailVerify', { email: values.email });
		} else {
			const data = await res.json();
			setLoading(false);
			setError(data.message);
			setInterval(() => {
				setError('');
			}, 10000);
		}
	};
	return (
		<TouchableWithoutFeedback onPress={Keyboard.dismiss}>
			<View style={styles.main}>
				<View style={styles.wrapper}>
					<Ant
						name='caretleft'
						size={30}
						color={'black'}
						onPress={() => navigation.navigate('Login')}
					/>
					<Text style={styles.text}>Forget Password</Text>
				</View>
				<View style={styles.emailwrapper}>
					<Text style={[styles.text, { marginLeft: '27%' }]}>
						ENTER EMAIL HERE
					</Text>
					<Text
						style={[
							styles.text,
							{
								marginLeft: '14%',
								color: 'grey',
								fontWeight: '300',
								fontSize: 12,
								marginTop: '4%',
							},
						]}>
						Enter the email associated with your account
					</Text>
				</View>
				<Formik
					validationSchema={emailValidationSchema}
					initialValues={{ email: '' }}
					onSubmit={values => main(values)}>
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
								disabled={!isValid || values.email === ''}>
								{loading ? (
									<ActivityIndicator
										size='small'
										color='#ffffff'
									/>
								) : (
									<Text style={styles.textsign}>Sign In</Text>
								)}
							</TouchableOpacity>
							{error !== '' ? (
								<Text
									style={{
										marginTop: 40,
										color: 'red',
										textAlign: 'center',
										fontWeight: 'bold',
										fontSize: 20,
									}}>
									{error}
								</Text>
							) : null}
						</View>
					)}
				</Formik>
			</View>
		</TouchableWithoutFeedback>
	);
};

const styles = StyleSheet.create({
	main: {
		flex: 1,
		padding: '3%',
	},
	text: {
		color: 'black',
		fontFamily: 'Poppins',
		fontSize: 20,
		fontWeight: '700',
		marginLeft: '20%',
	},
	wrapper: {
		flexDirection: 'row',
	},
	emailwrapper: {
		marginTop: '40%',
	},
	textinput: {
		backgroundColor: colors.lightOrange,
		marginTop: 10,
		fontSize: 13,
	},
	touchsign: {
		marginTop: '7%',
		backgroundColor: colors.primary,
		borderRadius: 40,
		height: '20%',
		justifyContent: 'center',
		alignItems: 'center',
	},
	textsign: {
		color: 'mintcream',
		fontFamily: 'Poppins',
		fontWeight: 'bold',
		fontSize: 16,
	},
	errorText: {
		fontSize: 10,
		color: 'red',
		alignSelf: 'flex-end',
	},
});

export default ForgotPassword;
