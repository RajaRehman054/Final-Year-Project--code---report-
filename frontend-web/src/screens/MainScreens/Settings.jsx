import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { ToastContainer, toast } from 'react-toastify';
import { editUser, changePass, changeNotification } from '../../api/Api';
import { setUser } from '../../redux/userSlice';
import UserOne from '../../assets/help.png';
import { IoCall, IoLocation, IoMail } from 'react-icons/io5';
const Settings = () => {
	const { user } = useSelector(state => state.user);
	const dispatch = useDispatch();
	const [activeTab, setactiveTab] = useState(0);

	const [name, setName] = useState(user.name);
	const [email, setEmail] = useState(user.email);
	const [address, setAddress] = useState(user.address);
	const [phone, setPhone] = useState(user.phone);

	const [oldPassword, setOldPassword] = useState('');
	const [newPassword, setNewPassword] = useState('');
	const [pass, setPass] = useState('');
	const [checkBox1, setCheckBox1] = useState(user.notification);
	const [checkBox2, setCheckBox2] = useState(user.notification_bid);
	const [checkBox3, setCheckBox3] = useState(user.notification_payment);

	const changeProfile = async () => {
		let payload = { name, email, address, phone };
		const res = await editUser(payload);
		if (res === 'Unauthorized') {
			return toast.error(`Email already associated.`, {
				position: 'bottom-right',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		dispatch(setUser(res.user));
		return toast.success(`Successfully edited profile.`, {
			position: 'bottom-right',
			autoClose: 5000,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
	};

	const handlePassword = async () => {
		if (newPassword !== pass) {
			return toast.error(`Password does not match.`, {
				position: 'bottom-right',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		let payload = {
			username: user.username,
			password: oldPassword,
			new_password: newPassword,
		};
		const res = await changePass(payload);
		if (res === 'Unauthorized') {
			return toast.error(`Please type correct old password.`, {
				position: 'bottom-right',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		return toast.success(`Password changed successfully.`, {
			position: 'bottom-right',
			autoClose: 5000,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
	};

	const handleNotification = async () => {
		let payload = {
			notification: checkBox1,
			notification_bid: checkBox2,
			notification_payment: checkBox3,
		};
		await changeNotification(payload);
		dispatch(
			setUser({
				...user,
				notification: checkBox1,
				notification_bid: checkBox2,
				notification_payment: checkBox3,
			})
		);
		return toast.success(`Notifications settings changed successfully.`, {
			position: 'bottom-right',
			autoClose: 5000,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
	};

	const renderActiveComponent = () => {
		switch (activeTab) {
			case 0:
				return (
					<div>
						<p className='text-[20px] font-bold'>Edit Profile</p>
						<div className='flex flex-row items-center my-[20px]'>
							<div className='w-[20%]'>
								<p className='text-[14px] font-bold '>Name</p>
							</div>

							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Full Name'
								value={name}
								onChange={e => setName(e.target.value)}
							/>
						</div>
						<div className='flex flex-row items-center my-[20px]'>
							<div className='w-[20%]'>
								<p className='text-[14px] font-bold '>
									Email address
								</p>
							</div>
							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Email Address'
								value={email}
								onChange={e => setEmail(e.target.value)}
							/>
						</div>
						<div className='flex flex-row items-center my-[20px]'>
							<div className='w-[20%]'>
								<p className='text-[14px] font-bold '>
									Phone Number
								</p>
							</div>
							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Phone No'
								value={phone}
								onChange={e => setPhone(e.target.value)}
							/>
						</div>
						<div className='flex flex-row items-center my-[20px]'>
							<div className='w-[20%]'>
								<p className='text-[14px] font-bold '>
									Address
								</p>
							</div>
							<input
								className='h-[60px] w-full  rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Address'
								value={address}
								onChange={e => setAddress(e.target.value)}
							/>
						</div>
						<div className=' flex flex-row  my-[20px] items-center justify-end'>
							<button
								className='mx-[10px] h-[22px] w-[100px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]  text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[100px]'
								style={{ transition: '0.3s' }}
								onClick={changeProfile}>
								Save
							</button>
							<button
								className='h-[22px] w-[100px] justify-center rounded-[10px] bg-[red]  text-[15px]   text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[100px]'
								style={{ transition: '0.3s' }}>
								Cancel
							</button>
						</div>
					</div>
				);
			case 1:
				return (
					<div>
						<p className='text-[20px] font-bold'>Edit Password</p>
						<div className='flex flex-row items-center my-[20px]'>
							<div className='w-[20%]'>
								<p className='text-[14px] font-bold '>
									Old Password
								</p>
							</div>
							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Old Password'
								value={oldPassword}
								onChange={e => setOldPassword(e.target.value)}
							/>
						</div>
						<div className='flex flex-row items-center my-[20px]'>
							<div className='w-[20%]'>
								<p className='text-[14px] font-bold '>
									New Password
								</p>
							</div>
							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='New Password'
								value={newPassword}
								onChange={e => setNewPassword(e.target.value)}
							/>
						</div>
						<div className='flex flex-row items-center my-[20px]'>
							<div className='w-[20%]'>
								<p className='text-[14px] font-bold '>
									Re-enter Password
								</p>
							</div>
							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Re-enter Password'
								value={pass}
								onChange={e => setPass(e.target.value)}
							/>
						</div>
						<div className=' flex flex-row  my-[20px] items-center justify-end'>
							<button
								className='mx-[10px] h-[22px] w-[100px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]  text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[100px]'
								style={{ transition: '0.3s' }}
								onClick={handlePassword}>
								Save
							</button>
							<button
								className='h-[22px] w-[100px] justify-center rounded-[10px] bg-[red]  text-[15px]   text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[100px]'
								style={{ transition: '0.3s' }}>
								Cancel
							</button>
						</div>
					</div>
				);
			case 3:
				return (
					<div>
						<p className='text-[20px] font-bold'>
							Edit Notifications
						</p>

						<div className='flex flex-row items-center my-[20px] h-[60px]'>
							<div className='w-[30%]'>
								<p className='text-[14px] font-bold '>
									Allow Notifications
								</p>
							</div>
							<input
								type='checkbox'
								className='toggle toggle-success ml-auto mr-[20px] '
								checked={checkBox1}
								onChange={() => setCheckBox1(!checkBox1)}
							/>
						</div>
						<div className='flex flex-row items-center my-[20px] h-[60px]'>
							<div className='w-[30%]'>
								<p className='text-[14px] font-bold '>
									Allow Bidding Updates
								</p>
							</div>
							<input
								type='checkbox'
								className='toggle toggle-success ml-auto mr-[20px]'
								checked={checkBox2}
								onChange={() => setCheckBox2(!checkBox2)}
							/>
						</div>
						<div className='flex flex-row items-center my-[20px] h-[60px]'>
							<div className='w-[30%]'>
								<p className='text-[14px] font-bold '>
									Allow Payment Updates
								</p>
							</div>
							<input
								type='checkbox'
								className='toggle toggle-success ml-auto mr-[20px]'
								checked={checkBox3}
								onChange={() => setCheckBox3(!checkBox3)}
							/>
						</div>
						<div className=' flex flex-row  my-[20px] items-center justify-end'>
							<button
								className='mx-[10px] h-[22px] w-[100px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]  text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[100px]'
								style={{ transition: '0.3s' }}
								onClick={handleNotification}>
								Save
							</button>
						</div>
					</div>
				);
			case 4:
				return (
					<div>
						<div
							className='bg-cover bg-center h-[550px] relative '
							style={{ backgroundImage: `url(${UserOne})` }}>
							<div className='absolute inset-0 bg-[black] bg-opacity-50 rounded-xl'></div>

							<div className='flex items-center flex-col justify-center relative z-10 h-[33%] '>
								<p className='font-[semibold] text-[50px] text-[white]'>
									How can we Help you?
								</p>
								<p className='font-[semibold] text-[20px] text-[white]'>
									we'll love to talk about how we can work
									together
								</p>
							</div>

							<div className=' flex items-center flex-row  justify-between relative z-10 h-[33%] '>
								<div className='card w-[30%] bg-white  '>
									<div className='card-body items-center justify-center'>
										<IoMail size={30} color='#FF8216' />
										<h2 className='card-title'>Email</h2>
										<p>newswave@gmail.com</p>
									</div>
								</div>
								<div className='card w-[30%] bg-white  '>
									<div className='card-body items-center justify-center'>
										<IoCall size={30} color='#FF8216' />
										<h2 className='card-title'>
											Phone Number
										</h2>
										<p>051-5212723</p>
									</div>
								</div>
							</div>
							<div className=' flex items-center flex-row justify-center relative z-10 h-[33%]'>
								<div className='card w-[30%] bg-white  '>
									<div className='card-body items-center justify-center'>
										<IoLocation size={30} color='#FF8216' />
										<h2 className='card-title'>
											Main Office
										</h2>
										<p>Comsats Islamabad</p>
									</div>
								</div>
							</div>
						</div>
					</div>
				);
		}
	};

	return (
		<>
			<div className='flex w-full flex-col'>
				{/*Top Heading */}
				<div className='flex w-full '>
					<h1 className='text-[20px] font-bold'>Settings</h1>
				</div>

				{/*View Area divided in two */}
				<div className='my-[22px] flex flex-wrap justify-between'>
					{/*First Column */}
					<div className='w-full sm:w-[75%]'>
						<div className='relative flex h-[60px] w-full items-center rounded-xl bg-white p-4 shadow-xl'>
							<div className='p-x-4 relative flex h-[40px]  w-full flex-row rounded-[10px]'>
								<div
									onClick={() => setactiveTab(0)}
									className={`${
										activeTab === 0
											? 'w-[70%] bg-[#FF8216] text-white'
											: 'w-[70%] bg-transparent text-[#939393]'
									} flex h-full cursor-pointer items-center justify-center rounded-[10px]`}>
									<p className='font-bold'>General</p>
								</div>
								<div
									onClick={() => setactiveTab(1)}
									className={`${
										activeTab === 1
											? 'w-[70%] bg-[#FF8216] text-white'
											: 'w-[70%] bg-transparent text-[#939393]'
									} flex h-full cursor-pointer items-center justify-center rounded-[10px] bg-[#FF8216]`}>
									<p className='font-bold'>Password</p>
								</div>
								<div
									onClick={() => setactiveTab(3)}
									className={`${
										activeTab === 3
											? 'w-[70%] bg-[#FF8216] text-white'
											: 'w-[70%] bg-transparent text-[#939393]'
									} flex h-full cursor-pointer items-center justify-center rounded-[10px]  bg-[#FF8216]`}>
									<p className='font-bold'>Notifications</p>
								</div>
								<div
									onClick={() => setactiveTab(4)}
									className={`${
										activeTab === 4
											? 'w-[70%] bg-[#FF8216] text-white'
											: 'w-[70%] bg-transparent text-[#939393]'
									} flex h-full cursor-pointer items-center justify-center rounded-[10px]  bg-[#FF8216]`}>
									<p className='font-bold'>Help</p>
								</div>
							</div>
						</div>

						{/* Jobs View */}
						<div className='relative my-[22px] flex w-full flex-col  rounded-xl bg-white p-4 shadow-xl'>
							{renderActiveComponent()}
						</div>
					</div>

					{/* Second Column View */}
					<div className='hidden w-full sm:block sm:w-[23%]'></div>
				</div>
			</div>
			<ToastContainer />
		</>
	);
};

export default Settings;
