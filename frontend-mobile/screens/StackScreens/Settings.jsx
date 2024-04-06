import React, { useState } from 'react';
import {
	Text,
	View,
	TouchableOpacity,
	StyleSheet,
	SafeAreaView,
	ScrollView,
} from 'react-native';

import Ionicons from 'react-native-vector-icons/Ionicons';
import { Switch } from 'react-native-paper';
import { useContext } from 'react';
import { AuthContext } from '../../auth/AuthContext';

export default function Settings({ navigation }) {
	const { logOut } = useContext(AuthContext);
	const [isEnabled, setIsEnabled] = useState(false);
	const [isEnabled1, setIsEnabled1] = useState(false);
	const [isEnabled2, setIsEnabled2] = useState(false);
	const [isEnabled3, setIsEnabled3] = useState(false);
	const [isEnabled4, setIsEnabled4] = useState(false);
	const [isEnabled5, setIsEnabled5] = useState(false);
	const [isEnabled6, setIsEnabled6] = useState(false);

	const [showDropdown, setShowDropdown] = useState(false);
	const [showDropdown1, setShowDropdown1] = useState(false);
	const [showDropdown2, setShowDropdown2] = useState(false);

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
								fontWeight: 'semibold',
								color: 'black',
							}}>
							Settings
						</Text>
					</View>
					<View>
						<Ionicons
							name='chevron-back'
							color={'white'}
							size={30}
						/>
					</View>
				</View>
			</View>

			<ScrollView>
				<View
					style={{
						height: 40,
						flexDirection: 'row',
						width: '93%',
						alignSelf: 'center',
					}}>
					<View>
						<Text style={styles.headingText}>Account</Text>
					</View>
				</View>
				<View style={styles.borderLine}></View>

				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='settings-sharp'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Basics</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name='chevron-forward'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='mail'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Change Email</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name='chevron-forward'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='lock-closed-sharp'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Change Password</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name='chevron-forward'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='location-sharp'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Change location</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name='chevron-forward'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='person-remove'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Blocked Users</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name='chevron-forward'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>

				<View
					style={{
						height: 40,
						flexDirection: 'row',
						width: '93%',
						alignSelf: 'center',
					}}>
					<View>
						<Text style={styles.headingText}>Manage</Text>
					</View>
				</View>
				<View style={styles.borderLine}></View>
				<TouchableOpacity
					onPress={() => setShowDropdown(!showDropdown)}
					style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='videocam'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Video Autoplay</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name={showDropdown ? 'chevron-up' : 'chevron-down'}
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				{showDropdown && (
					<View>
						<View
							style={[
								styles.button11,
								{ width: '85%', alignSelf: 'center' },
							]}>
							<View style={{ width: '10%' }}></View>
							<Text>Autoplay</Text>
							<View style={styles.togglebutton}>
								<Switch
									style={{
										marginLeft: 'auto',
										marginRight: 'auto',
									}}
									trackColor={{
										false: 'lightgray',
										true: 'black',
									}}
									thumbColor={isEnabled ? 'white' : 'black'}
									onValueChange={() =>
										setIsEnabled(!isEnabled)
									}
									value={isEnabled}
								/>
							</View>
						</View>
					</View>
				)}
				<TouchableOpacity
					onPress={() => setShowDropdown1(!showDropdown1)}
					style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='notifications'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Notification</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name={showDropdown1 ? 'chevron-up' : 'chevron-down'}
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				{showDropdown1 && (
					<View>
						<View
							style={[
								styles.button11,
								{ width: '85%', alignSelf: 'center' },
							]}>
							<View style={{ width: '10%' }}></View>
							<Text>Jobs</Text>
							<View style={styles.togglebutton}>
								<Switch
									style={{
										marginLeft: 'auto',
										marginRight: 'auto',
									}}
									trackColor={{
										false: 'lightgray',
										true: 'black',
									}}
									thumbColor={isEnabled3 ? 'white' : 'black'}
									onValueChange={() =>
										setIsEnabled3(!isEnabled3)
									}
									value={isEnabled3}
								/>
							</View>
						</View>

						<View
							style={[
								styles.button11,
								{ width: '85%', alignSelf: 'center' },
							]}>
							<View style={{ width: '10%' }}></View>
							<Text>Chats</Text>
							<View style={styles.togglebutton}>
								<Switch
									style={{
										marginLeft: 'auto',
										marginRight: 'auto',
									}}
									trackColor={{
										false: 'lightgray',
										true: 'black',
									}}
									thumbColor={isEnabled4 ? 'white' : 'black'}
									onValueChange={() =>
										setIsEnabled4(!isEnabled4)
									}
									value={isEnabled4}
								/>
							</View>
						</View>

						<View
							style={[
								styles.button11,
								{ width: '85%', alignSelf: 'center' },
							]}>
							<View style={{ width: '10%' }}></View>
							<Text>Follower</Text>
							<View style={styles.togglebutton}>
								<Switch
									style={{
										marginLeft: 'auto',
										marginRight: 'auto',
									}}
									trackColor={{
										false: 'lightgray',
										true: 'black',
									}}
									thumbColor={isEnabled5 ? 'white' : 'black'}
									onValueChange={() =>
										setIsEnabled5(!isEnabled5)
									}
									value={isEnabled5}
								/>
							</View>
						</View>

						<View
							style={[
								styles.button11,
								{ width: '85%', alignSelf: 'center' },
							]}>
							<View style={{ width: '10%' }}></View>
							<Text>Comments</Text>
							<View style={styles.togglebutton}>
								<Switch
									style={{
										marginLeft: 'auto',
										marginRight: 'auto',
									}}
									trackColor={{
										false: 'lightgray',
										true: 'black',
									}}
									thumbColor={isEnabled6 ? 'white' : 'black'}
									onValueChange={() =>
										setIsEnabled6(!isEnabled6)
									}
									value={isEnabled6}
								/>
							</View>
						</View>
					</View>
				)}
				<TouchableOpacity
					onPress={() => setShowDropdown2(!showDropdown2)}
					style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='alert'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Jobs & Chats</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name={showDropdown2 ? 'chevron-up' : 'chevron-down'}
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				{showDropdown2 && (
					<View>
						<View
							style={[
								styles.button11,
								{ width: '85%', alignSelf: 'center' },
							]}>
							<View style={{ width: '10%' }}></View>
							<Text>Recieve Jobs</Text>
							<View style={styles.togglebutton}>
								<Switch
									style={{
										marginLeft: 'auto',
										marginRight: 'auto',
									}}
									trackColor={{
										false: 'lightgray',
										true: 'black',
									}}
									thumbColor={isEnabled1 ? 'white' : 'black'}
									onValueChange={() =>
										setIsEnabled1(!isEnabled1)
									}
									value={isEnabled1}
								/>
							</View>
						</View>
						<View
							style={[
								styles.button11,
								{ width: '85%', alignSelf: 'center' },
							]}>
							<View style={{ width: '10%' }}></View>
							<Text>Recieve Chats</Text>
							<View style={styles.togglebutton}>
								<Switch
									style={{
										marginLeft: 'auto',
										marginRight: 'auto',
									}}
									trackColor={{
										false: 'lightgray',
										true: 'black',
									}}
									thumbColor={isEnabled2 ? 'white' : 'black'}
									onValueChange={() =>
										setIsEnabled2(!isEnabled2)
									}
									value={isEnabled2}
								/>
							</View>
						</View>
					</View>
				)}

				<View
					style={{
						height: 40,
						flexDirection: 'row',
						width: '93%',
						alignSelf: 'center',
					}}>
					<View>
						<Text style={styles.headingText}>Support</Text>
					</View>
				</View>
				<View style={styles.borderLine}></View>

				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='document-text-sharp'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Terms and conditions</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name='chevron-forward'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='shield-checkmark-sharp'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Privacy Policy</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name='chevron-forward'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='clipboard'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text style={styles.text12}>Feedback</Text>
					<View style={{ width: '15%', marginLeft: 'auto' }}>
						<Ionicons
							name='chevron-forward'
							color={'black'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
				</TouchableOpacity>
				<View style={{ height: 20, flexDirection: 'row' }}></View>
				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='archive'
							color={'#00bfff'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text
						style={{
							color: '#00bfff',
							fontWeight: 'bold',
							fontSize: 16,
						}}>
						Clear Search History
					</Text>
				</TouchableOpacity>
				<TouchableOpacity style={styles.button11}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='trash'
							color={'red'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text
						style={{
							color: 'red',
							fontWeight: 'bold',
							fontSize: 16,
						}}>
						Delete Account
					</Text>
				</TouchableOpacity>
				<TouchableOpacity style={styles.button11} onPress={logOut}>
					<View style={{ width: '15%' }}>
						<Ionicons
							name='log-out'
							color={'red'}
							size={25}
							style={{ alignSelf: 'center' }}
						/>
					</View>
					<Text
						style={{
							color: 'red',
							fontWeight: 'bold',
							fontSize: 16,
						}}>
						Log Out
					</Text>
				</TouchableOpacity>

				<View
					style={{
						height: 60,
						alignItems: 'center',
						justifyContent: 'center',
					}}>
					<View>
						<Text
							style={{
								color: 'black',
								fontWeight: 'bold',
								fontStyle: 'italic',
							}}>
							NewsWave V5.11
						</Text>
					</View>
				</View>
			</ScrollView>
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
		width: '95%',
		height: '100%',
		flexDirection: 'row',
		alignItems: 'center',
		marginTop: 'auto',
		justifyContent: 'space-between',
	},

	text12: {
		color: 'black',
		fontWeight: 'medium',
		fontSize: 16,
	},
	button11: {
		flexDirection: 'row',
		alignItems: 'center',
		backgroundColor: 'white',
		height: 50,
	},
	headingText: {
		marginTop: 'auto',
		marginBottom: 5,
		color: 'darkgray',
		fontWeight: 'bold',
		fontSize: 18,
	},
	togglebutton: { width: '15%', marginLeft: 'auto', marginRight: 10 },
	borderLine: {
		width: '93%',
		alignSelf: 'center',
		borderColor: 'darkgray',
		borderBottomWidth: 1,
	},
});
