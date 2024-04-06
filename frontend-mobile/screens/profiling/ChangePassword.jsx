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

import confirmPasswordValidationSchema from '../../utils/validations/confirmPasswordValidation';
import { colors } from '../../utils/theme.js';

const ChangePassword = ({ navigation, route }) => {
	const [error, setError] = useState('');
	const [loading, setLoading] = useState(false);
	const [show, setShow] = useState(false);
	const { email } = route.params;

	const main = async values => {
		console.log(email);
		setLoading(true);
		const opts = {
			method: 'PATCH',
			headers: {
				'Content-type': 'application/json',
			},
			body: JSON.stringify({
				email,
				password: values.password,
			}),
		};
		const res = await fetch(`${url}/users/passwordreset`, opts);
		if (res.ok) {
			Toast.show({
				type: 'success',
				text1: 'Password Reset Successfully. ✅',
			});
			navigation.navigate('Login');
		} else {
			const data = await res.json();
			setError(data.message);
			setLoading(false);
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
					<Text style={styles.text}>Reset Password</Text>
				</View>
				<View style={styles.emailwrapper}>
					<Text
						style={[
							styles.text,
							{ marginLeft: '23%', fontSize: 30 },
						]}>
						New Password
					</Text>
					<Text
						style={[
							styles.text,
							{
								marginLeft: '8%',
								color: 'grey',
								fontWeight: '300',
								fontSize: 12,
								marginTop: '4%',
							},
						]}>
						New Password must be different from old Password
					</Text>
				</View>
				<Formik
					validationSchema={confirmPasswordValidationSchema}
					initialValues={{ password: '', confirmPassword: '' }}
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
								value={values.confirmpassword}
								onChangeText={handleChange('confirmPassword')}
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
								disabled={!isValid || values.email === ''}>
								{loading ? (
									<ActivityIndicator
										size='small'
										color='#ffffff'
									/>
								) : (
									<Text style={styles.textsign}>
										Reset Password
									</Text>
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
		height: '17%',
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

export default ChangePassword;
