import React, { useState } from 'react';
import {
	Text,
	View,
	TouchableOpacity,
	StyleSheet,
	SafeAreaView,
	TextInput,
} from 'react-native';

import Ionicons from 'react-native-vector-icons/Ionicons';

export default function Teleprompter({ navigation }) {
	const [teleText, setteleText] = useState('');

	return (
		<SafeAreaView style={{ backgroundColor: 'white', flex: 1 }}>
			<View style={styles.uppertab}>
				<View style={styles.innertab}>
					<TouchableOpacity onPress={() => navigation.goBack()}>
						<Ionicons
							name='chevron-back'
							color={'black'}
							size={30}
						/>
					</TouchableOpacity>
					<View>
						<Text
							style={{
								fontSize: 20,
								fontWeight: 'medium',
								color: 'black',
							}}>
							Teleprompter
						</Text>
					</View>
					<TouchableOpacity
						onPress={() =>
							navigation.navigate('CameraRecording', {
								text: teleText,
							})
						}>
						<Text
							style={{
								fontSize: 18,
								fontWeight: 'medium',
								color: '#0f89da',
							}}>
							Save
						</Text>
					</TouchableOpacity>
				</View>
			</View>

			<View style={{ width: '90%', alignSelf: 'center' }}>
				<View style={{ marginTop: 10 }}>
					<TextInput
						style={{
							padding: 10,
							fontSize: 20,
							height: 100,
							color: 'black',
						}}
						placeholder={'Write your script here'}
						placeholderTextColor='darkgray'
						multiline={true}
						numberOfLines={4}
						value={teleText}
						onChangeText={setteleText}
					/>
				</View>
			</View>
		</SafeAreaView>
	);
}

const styles = StyleSheet.create({
	uppertab: {
		height: 70,
		backgroundColor: 'white',
		marginBottom: 5,
		alignItems: 'center',
		shadowColor: '#171717',
		shadowOffset: { width: -2, height: 4 },
		shadowOpacity: 0.2,
		shadowRadius: 3,
		elevation: 10,
	},
	innertab: {
		width: '92%',
		height: '100%',
		flexDirection: 'row',
		alignItems: 'center',
		marginTop: 'auto',
		justifyContent: 'space-between',
	},
});
