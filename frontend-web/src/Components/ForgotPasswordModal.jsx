import React, { useState } from 'react';
import { otpGenerateApi, verifyOtp, resetPasswordApi } from '../api/Api';
import { toast } from 'react-toastify';
import Loader from '../Components/Loader/Loader';

const ForgotPasswordModal = ({ handleModal }) => {
	const [activeTab, setactiveTab] = useState(0);
	const [email, setemail] = useState('');
	const [pinCode, setPinCode] = useState('');
	const [password, setPassword] = useState('');
	const [resetPassword, setresetPassword] = useState('');

	const getOtp = async () => {
		if (email === '') {
			return toast.error(`Please enter a valid email`, {
				position: 'top-center',
				autoClose: 2000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		setactiveTab(4);
		const res = await otpGenerateApi(email);
		if (res?.status === 404) {
			setactiveTab(0);
			return toast.error(`Email does not exist`, {
				position: 'top-center',
				autoClose: 2000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		setactiveTab(1);
	};

	const verifyCode = async () => {
		let payload = { email, otp: pinCode };
		if (pinCode === '') {
			return toast.error(`Please enter 4 digit code.`, {
				position: 'top-center',
				autoClose: 2000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		const res = await verifyOtp(payload);
		if (res?.message === 'Invalid or Expired token') {
			return toast.error(`${res.message}`, {
				position: 'top-center',
				autoClose: 2000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		setactiveTab(2);
	};

	const resetPasswordFunction = async () => {
		let payload = { email, password };
		if (password === '' && resetPassword == '') {
			return toast.error(`Please fill all the fields.`, {
				position: 'top-center',
				autoClose: 2000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		await resetPasswordApi(payload);
		setactiveTab(0);
		handleModal(true);
		return toast.success(`Password changed successfully`, {
			position: 'top-center',
			autoClose: 2000,
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
						<h3 className='text-center text-lg font-bold text-[#FF8216]'>
							Enter Email
						</h3>
						<p className='pt-4 text-[black]'>
							Please enter your email to recieve an Otp Code:
						</p>
						<div className='mt-[30px] flex w-full'>
							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Email'
								value={email}
								onChange={e => setemail(e.target.value)}
							/>
						</div>
						<div className='mt-[20px] flex w-full'>
							<button
								className='h-[60px] w-full rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]'
								style={{ transition: '0.3s' }}
								onClick={getOtp}>
								Next
							</button>
						</div>
					</div>
				);
			case 1:
				return (
					<div>
						<h3 className='text-center text-lg font-bold text-[#FF8216]'>
							Verify
						</h3>
						<p className='pt-4 text-[black]'>
							We have sent a pin to your email :
							<span className='font-semibold text-[#FF8216]'>
								{email}
							</span>
						</p>
						<p className='pt-4 text-[black]'>
							Please enter the pin to verify your account
						</p>
						<div className='mt-[30px] flex w-full'>
							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='4 Digit Pin Code'
								value={pinCode}
								onChange={e => setPinCode(e.target.value)}
							/>
						</div>
						<div className='mt-[20px] flex w-full'>
							<button
								className='h-[60px] w-full rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]'
								style={{ transition: '0.3s' }}
								onClick={verifyCode}>
								Verify
							</button>
						</div>
						<div className='mt-[20px] flex w-full  justify-center '>
							<p className='font-regular text-[15px] text-[black]'>
								Get Verification Code?
							</p>
							<button
								className='text-button ml-[5px] text-[15px] font-semibold text-[#FF8216] hover:text-blue-700 focus:outline-none'
								onClick={getOtp}>
								Resend
							</button>
						</div>
					</div>
				);
			case 2:
				return (
					<div>
						<h3 className='text-center text-lg font-bold text-[#FF8216]'>
							Reset Password
						</h3>

						<div className='mt-[30px] flex w-full'>
							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Enter New Password'
								value={password}
								onChange={e => setPassword(e.target.value)}
							/>
						</div>
						<div className='mt-[30px] flex w-full'>
							<input
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Re-enter Password'
								value={resetPassword}
								onChange={e => setresetPassword(e.target.value)}
							/>
						</div>
						<div className='mt-[20px] flex w-full'>
							<button
								className='h-[60px] w-full rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]'
								style={{ transition: '0.3s' }}
								onClick={resetPasswordFunction}>
								Reset
							</button>
						</div>
					</div>
				);
			case 4:
				return (
					<div class='flex items-center justify-center '>
						<span className='loading loading-spinner loading-md m-auto'></span>
					</div>
				);
		}
	};

	return (
		<dialog id='my_modal' className='modal '>
			<div className='modal-box w-[400px] bg-white'>
				<form method='dialog'>
					{/* if there is a button in form, it will close the modal */}
					<button className='btn btn-circle btn-ghost btn-sm absolute right-2 top-2'>
						✕
					</button>
				</form>
				{renderActiveComponent()}
			</div>
		</dialog>
	);
};

export default ForgotPasswordModal;
