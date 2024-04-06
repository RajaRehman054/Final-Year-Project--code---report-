import {
	View,
	StyleSheet,
	Text,
	TouchableOpacity,
	TouchableWithoutFeedback,
	Keyboard,
	ActivityIndicator,
} from 'react-native';
import { useState } from 'react';
import Ant from 'react-native-vector-icons/AntDesign';
import { TextInput } from 'react-native-paper';
import Toast from 'react-native-toast-message';

import { colors } from '../../utils/theme.js';
import url from '../../utils/url.js';

const EmailVerify = ({ navigation, route }) => {
	const [box1, setBox1] = useState('');
	const [box2, setBox2] = useState('');
	const [box3, setBox3] = useState('');
	const [box4, setBox4] = useState('');
	const [error, setError] = useState('');
	const [loading, setLoading] = useState(false);
	const { email } = route.params;

	const handleSubmit = async () => {
		setLoading(true);
		const otp = box1 + box2 + box3 + box4;
		const res = await fetch(`${url}/users/otpVerify/${email}/${otp}`);
		if (res.ok) {
			navigation.navigate('ChangePassword', { email });
		} else {
			const data = await res.json();
			setError(data.message);
			setLoading(false);
			setInterval(() => {
				setError('');
			}, 10000);
		}
	};

	const resendCode = async () => {
		const res = await fetch(`${url}/users/otp/${email}`);
		if (res.ok) {
			Toast.show({
				type: 'success',
				text1: 'Code is sent again. ✅',
			});
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
						onPress={() => navigation.navigate('ForgotPassword')}
					/>
					<Text style={styles.text}>Email Verification</Text>
				</View>
				<View style={styles.emailwrapper}>
					<Text style={[styles.text, { marginLeft: '27%' }]}>
						GET YOUR CODE
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
						Enter the 4-digits code send to your email address
					</Text>
				</View>
				<View style={styles.mainwrapper}>
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
						value={box1}
						onChangeText={setBox1}
						keyboardType='numeric'
						maxLength={1}
					/>
					<TextInput
						outlineColor={colors.primary}
						theme={{
							colors: {
								text: 'black',
								primary: colors.orange,
							},
							roundness: 100,
						}}
						style={styles.textinput}
						mode='outlined'
						value={box2}
						onChangeText={setBox2}
						keyboardType='numeric'
						maxLength={1}
					/>
					<TextInput
						outlineColor={colors.primary}
						theme={{
							colors: {
								text: 'black',
								primary: colors.orange,
							},
							roundness: 100,
						}}
						style={styles.textinput}
						mode='outlined'
						value={box3}
						onChangeText={setBox3}
						keyboardType='numeric'
						maxLength={1}
					/>
					<TextInput
						outlineColor={colors.primary}
						theme={{
							colors: {
								text: 'black',
								primary: colors.orange,
							},
							roundness: 100,
						}}
						style={styles.textinput}
						mode='outlined'
						value={box4}
						onChangeText={setBox4}
						keyboardType='numeric'
						maxLength={1}
					/>
				</View>
				<TouchableOpacity
					style={[
						styles.touchsign,
						{
							backgroundColor:
								box1 === '' ||
								box2 === '' ||
								box3 === '' ||
								box4 === ''
									? '#FFA559'
									: colors.primary,
						},
					]}
					onPress={handleSubmit}
					disabled={
						box1 === '' || box2 === '' || box3 === '' || box4 === ''
							? true
							: false
					}>
					{loading ? (
						<ActivityIndicator size='small' color='#ffffff' />
					) : (
						<Text style={styles.textsign}>Verify</Text>
					)}
				</TouchableOpacity>
				<TouchableOpacity
					style={styles.touchbottom}
					onPress={resendCode}>
					<Text style={styles.textbottom}>
						Didn't Receive a code?
						<Text style={styles.textc}>Resend Code</Text>
					</Text>
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
	mainwrapper: {
		flexDirection: 'row',
		justifyContent: 'space-evenly',
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
		textAlign: 'center',
	},
	touchsign: {
		marginTop: '7%',
		backgroundColor: colors.primary,
		borderRadius: 40,
		height: '7%',
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
	textc: {
		color: colors.primary,
	},
	textbottom: {
		color: 'black',
	},
	touchbottom: {
		justifyContent: 'center',
		alignItems: 'center',
		marginTop: '5%',
	},
});

export default EmailVerify;
