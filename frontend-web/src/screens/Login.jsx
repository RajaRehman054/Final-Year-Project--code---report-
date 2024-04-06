import { useState } from 'react';
import { useDispatch } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import Pic from '.././assets/orange-logo.png';
import { loginApi, registerApi } from '../api/Api';
import { ToastContainer, toast } from 'react-toastify';
import { setToken } from '../redux/tokenSlice.js';
import ForgotPasswordModal from '../Components/ForgotPasswordModal';

const handleModal = async checker => {
	if (checker?.bubbles === true) {
		return document.getElementById('my_modal').showModal();
	}
	document.getElementById('my_modal').close();
};

const Login = () => {
	// States
	const navigate = useNavigate();
	const dispatch = useDispatch();
	const [username, setUsername] = useState('');
	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');
	const [confirmPass, setConfirmPass] = useState('');
	const [checked, setChecked] = useState(false);
	const [showPassword, setShowPassword] = useState(false);
	const [isContent1Visible, setContent1Visible] = useState(false);
	const [loading, setLoading] = useState(false);

	const toggleContent = () => {
		setContent1Visible(!isContent1Visible);
	};

	const isValidEmail = input => {
		const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
		return emailRegex.test(input);
	};

	// Handle Login
	const handleLogin = async () => {
		if (username === '' || password === '') {
			return toast.warn('Please fill all the fields!', {
				position: 'top-center',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		setLoading(true);
		const res = await loginApi({ username, password });
		if (res === 'Unauthorized') {
			setLoading(false);
			return toast.error(`Incorrect Credentials!`, {
				position: 'top-center',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		if (checked) {
			localStorage.setItem('jwt', res.token);
		} else {
			dispatch(setToken(res.token));
		}
		navigate('/', { replace: true });
		setLoading(false);
	};

	//handle Signup
	const handleSignUp = async () => {
		if (
			username === '' ||
			password === '' ||
			confirmPass === '' ||
			email === ''
		) {
			return toast.warn('Please fill all the fields!', {
				position: 'top-center',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		if (!isValidEmail(email)) {
			return toast.warn('Please enter a valid email', {
				position: 'top-center',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		if (!checked) {
			return toast.warn('Please check Terms and Policy.', {
				position: 'top-center',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		setLoading(true);
		let payload = { username, email, password };
		const data = await registerApi(payload);
		if (data.status === 409) {
			setLoading(false);
			return toast.error(`${data.message}!`, {
				position: 'top-center',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		setLoading(false);
		setContent1Visible(false);
		toast.success('User Created Successfully', {
			position: 'top-center',
			autoClose: 5000,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
	};

	// switchingView
	const switchingView = () => {
		if (isContent1Visible) {
			return (
				<div className='order-2 flex w-full flex-col items-center justify-center bg-sidebar py-[15px] sm:w-[50%] md:order-2 md:w-[50%] md:shadow-md '>
					{/* Logo Image */}
					<div>
						<img src={Pic} alt='My Image' />
					</div>
					<p className='mt-[5px] text-[20px] font-semibold text-[black]'>
						Create New Account on
						<span className='text-[#FF8216]'>NewsWave</span>
					</p>

					{/* username */}
					<div className='mt-[40px] flex w-[60%] justify-center '>
						<input
							className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
							placeholder='Username'
							value={username}
							onChange={e => setUsername(e.target.value)}
						/>
					</div>

					{/* Email */}
					<div className='mt-[12px] flex w-[60%] justify-center '>
						<input
							type='email'
							className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
							placeholder='Email'
							value={email}
							onChange={e => setEmail(e.target.value)}
						/>
					</div>

					{/* Password */}
					<div className='mt-[12px] flex w-[60%]  justify-center '>
						<div className='relative w-full'>
							<input
								type={showPassword ? 'text' : 'password'}
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Password'
								value={password}
								onChange={e => setPassword(e.target.value)}
								hidden={false}
							/>
							<div
								className='absolute right-0 top-0 flex h-full cursor-pointer items-center justify-center pr-[22px]'
								onClick={() => setShowPassword(!showPassword)}>
								{!showPassword ? (
									<svg
										width='28'
										height='18'
										viewBox='0 0 28 18'
										fill='none'
										xmlns='http://www.w3.org/2000/svg'>
										<path
											d='M27.83 8.45C20.1374 -2.53655 7.82641 -2.493 0.169959 8.45025C0.0631809 8.61337 0.00547689 8.80371 0.00370422 8.99866C0.00193154 9.19361 0.0561647 9.38497 0.159959 9.55C7.83312 20.5365 20.2485 20.5038 27.84 9.53974C27.9437 9.37653 27.9979 9.18677 27.9962 8.99339C27.9944 8.80001 27.9367 8.61128 27.83 8.45ZM14 14.44C12.5576 14.4385 11.1748 13.8649 10.155 12.845C9.13509 11.8251 8.56146 10.4423 8.55996 9C8.81487 1.8 19.1861 1.80186 19.44 9.00009C19.4384 10.4424 18.8648 11.8252 17.8449 12.845C16.8251 13.8649 15.4423 14.4385 14 14.44Z'
											fill='#FF8216'
										/>
										<path
											d='M14 5.56C13.0929 5.56789 12.2255 5.9338 11.5869 6.57808C10.9482 7.22235 10.5898 8.09283 10.5898 9.00003C10.5899 9.90723 10.9482 10.7777 11.5869 11.422C12.2256 12.0662 13.0929 12.4321 14.0001 12.44C14.9073 12.4321 15.7746 12.0662 16.4133 11.4219C17.052 10.7776 17.4103 9.90716 17.4103 8.99996C17.4103 8.09276 17.0519 7.22229 16.4132 6.57803C15.7745 5.93377 14.9072 5.56788 14 5.56Z'
											fill='#FF8216'
										/>
									</svg>
								) : (
									<svg
										width='30'
										height='22'
										viewBox='0 0 30 22'
										fill='none'
										xmlns='http://www.w3.org/2000/svg'>
										<path
											fillRule='evenodd'
											clipRule='evenodd'
											d='M24.2928 0.2925L4.34577 20.2395C3.95577 20.6295 3.95577 21.2625 4.34577 21.6535C4.73677 22.0435 5.36977 22.0435 5.75977 21.6535L25.7068 1.7065C26.0968 1.3165 26.0968 0.6825 25.7068 0.2925C25.3168 -0.0975 24.6828 -0.0975 24.2928 0.2925ZM23.8578 4.9695L19.6548 9.1725C19.8778 9.7385 19.9998 10.3545 19.9998 10.9995C19.9998 13.7595 17.7598 15.9995 14.9998 15.9995C14.3548 15.9995 13.7388 15.8775 13.1728 15.6545L10.2148 18.6125C11.7168 19.1575 13.3268 19.4995 14.9998 19.4995C23.1418 19.4995 29.7768 11.3925 29.7768 11.3925C30.0738 11.0255 30.0738 10.5005 29.7768 10.1335C29.7768 10.1335 27.4478 7.2885 23.8578 4.9695ZM6.43076 16.7405L10.3448 12.8265C10.1218 12.2605 9.99976 11.6445 9.99976 10.9995C9.99976 8.2395 12.2398 5.9995 14.9998 5.9995C15.6448 5.9995 16.2608 6.1215 16.8268 6.3445L20.1288 3.0425C18.5298 2.4235 16.8018 2.02649 14.9998 2.02649C6.85776 2.02649 0.22275 10.1335 0.22275 10.1335C-0.07425 10.5005 -0.07425 11.0255 0.22275 11.3925C0.22275 11.3925 2.67676 14.3905 6.43076 16.7405ZM17.9948 10.8325C17.9978 10.8875 17.9998 10.9435 17.9998 10.9995C17.9998 12.6555 16.6558 13.9995 14.9998 13.9995C14.9438 13.9995 14.8878 13.9975 14.8328 13.9945L17.9948 10.8325ZM12.0048 11.1665C12.0018 11.1115 11.9998 11.0555 11.9998 10.9995C11.9998 9.3435 13.3438 7.9995 14.9998 7.9995C15.0558 7.9995 15.1117 8.00151 15.1667 8.00451L12.0048 11.1665Z'
											fill='#FF8216'
										/>
									</svg>
								)}
							</div>
						</div>
					</div>

					{/* confirm Password */}
					<div className='mt-[12px] flex w-[60%]  justify-center '>
						<div className='relative w-full'>
							<input
								type={showPassword ? 'text' : 'password'}
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Confirm Password'
								value={confirmPass}
								onChange={e => setConfirmPass(e.target.value)}
								hidden={false}
							/>
							<div
								className='absolute right-0 top-0 flex h-full cursor-pointer items-center justify-center pr-[22px]'
								onClick={() => setShowPassword(!showPassword)}>
								{!showPassword ? (
									<svg
										width='28'
										height='18'
										viewBox='0 0 28 18'
										fill='none'
										xmlns='http://www.w3.org/2000/svg'>
										<path
											d='M27.83 8.45C20.1374 -2.53655 7.82641 -2.493 0.169959 8.45025C0.0631809 8.61337 0.00547689 8.80371 0.00370422 8.99866C0.00193154 9.19361 0.0561647 9.38497 0.159959 9.55C7.83312 20.5365 20.2485 20.5038 27.84 9.53974C27.9437 9.37653 27.9979 9.18677 27.9962 8.99339C27.9944 8.80001 27.9367 8.61128 27.83 8.45ZM14 14.44C12.5576 14.4385 11.1748 13.8649 10.155 12.845C9.13509 11.8251 8.56146 10.4423 8.55996 9C8.81487 1.8 19.1861 1.80186 19.44 9.00009C19.4384 10.4424 18.8648 11.8252 17.8449 12.845C16.8251 13.8649 15.4423 14.4385 14 14.44Z'
											fill='#FF8216'
										/>
										<path
											d='M14 5.56C13.0929 5.56789 12.2255 5.9338 11.5869 6.57808C10.9482 7.22235 10.5898 8.09283 10.5898 9.00003C10.5899 9.90723 10.9482 10.7777 11.5869 11.422C12.2256 12.0662 13.0929 12.4321 14.0001 12.44C14.9073 12.4321 15.7746 12.0662 16.4133 11.4219C17.052 10.7776 17.4103 9.90716 17.4103 8.99996C17.4103 8.09276 17.0519 7.22229 16.4132 6.57803C15.7745 5.93377 14.9072 5.56788 14 5.56Z'
											fill='#FF8216'
										/>
									</svg>
								) : (
									<svg
										width='30'
										height='22'
										viewBox='0 0 30 22'
										fill='none'
										xmlns='http://www.w3.org/2000/svg'>
										<path
											fillRule='evenodd'
											clipRule='evenodd'
											d='M24.2928 0.2925L4.34577 20.2395C3.95577 20.6295 3.95577 21.2625 4.34577 21.6535C4.73677 22.0435 5.36977 22.0435 5.75977 21.6535L25.7068 1.7065C26.0968 1.3165 26.0968 0.6825 25.7068 0.2925C25.3168 -0.0975 24.6828 -0.0975 24.2928 0.2925ZM23.8578 4.9695L19.6548 9.1725C19.8778 9.7385 19.9998 10.3545 19.9998 10.9995C19.9998 13.7595 17.7598 15.9995 14.9998 15.9995C14.3548 15.9995 13.7388 15.8775 13.1728 15.6545L10.2148 18.6125C11.7168 19.1575 13.3268 19.4995 14.9998 19.4995C23.1418 19.4995 29.7768 11.3925 29.7768 11.3925C30.0738 11.0255 30.0738 10.5005 29.7768 10.1335C29.7768 10.1335 27.4478 7.2885 23.8578 4.9695ZM6.43076 16.7405L10.3448 12.8265C10.1218 12.2605 9.99976 11.6445 9.99976 10.9995C9.99976 8.2395 12.2398 5.9995 14.9998 5.9995C15.6448 5.9995 16.2608 6.1215 16.8268 6.3445L20.1288 3.0425C18.5298 2.4235 16.8018 2.02649 14.9998 2.02649C6.85776 2.02649 0.22275 10.1335 0.22275 10.1335C-0.07425 10.5005 -0.07425 11.0255 0.22275 11.3925C0.22275 11.3925 2.67676 14.3905 6.43076 16.7405ZM17.9948 10.8325C17.9978 10.8875 17.9998 10.9435 17.9998 10.9995C17.9998 12.6555 16.6558 13.9995 14.9998 13.9995C14.9438 13.9995 14.8878 13.9975 14.8328 13.9945L17.9948 10.8325ZM12.0048 11.1665C12.0018 11.1115 11.9998 11.0555 11.9998 10.9995C11.9998 9.3435 13.3438 7.9995 14.9998 7.9995C15.0558 7.9995 15.1117 8.00151 15.1667 8.00451L12.0048 11.1665Z'
											fill='#FF8216'
										/>
									</svg>
								)}
							</div>
						</div>
					</div>

					{/* Remember me */}
					<div className='mt-[18px] flex w-[60%]  cursor-pointer select-none flex-row items-center gap-x-2 '>
						<div
							onClick={() => setChecked(!checked)}
							className='h-[17px] w-[25px] rounded-[100px] border-[2px] border-[#FF8216]'>
							{checked && (
								<div className='flex h-full w-full items-center justify-center rounded-[50px] bg-[#FF8216]'>
									<svg
										width='13'
										height='8'
										viewBox='0 0 13 8'
										fill='none'
										xmlns='http://www.w3.org/2000/svg'>
										<path
											d='M2.80984 6.74445C2.88584 6.86721 3.00861 6.96075 3.15476 6.98998C3.18984 6.99583 3.22491 7.00167 3.25999 7.00167C3.37107 7.00167 3.4763 6.9666 3.56399 6.90229L11.883 0.956819C12.1168 0.787282 12.1753 0.459902 12.0057 0.220212C11.8362 -0.0136315 11.5088 -0.0720928 11.2691 0.0974438L3.41199 5.70969L0.980016 1.68173C0.828018 1.43035 0.506482 1.35435 0.255101 1.5005C0.00371878 1.6525 -0.0722798 1.97404 0.0738724 2.22542L2.80984 6.74445Z'
											fill='white'
										/>
									</svg>
								</div>
							)}
						</div>
						<div>
							<p className='text-[#48525B]'>
								I have read and agreed to Newswave
								<span className='text-[#FF8216]'>
									Terms of use, Privacy policy,
								</span>
								and
								<span className='text-[#FF8216]'>
									Disclaimer
								</span>
							</p>
						</div>
					</div>

					{/* Login Button */}
					<div className='mt-[30px] flex w-[60%]  justify-center'>
						<button
							onClick={handleSignUp}
							className='h-[60px] w-full rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]'
							style={{ transition: '0.3s' }}>
							{loading ? (
								<span className='loading loading-spinner text-accent'></span>
							) : (
								<p>Sign Up</p>
							)}
						</button>
					</div>

					{/* SignUp Option */}
					<div className='mt-[20px] flex w-[60%]  justify-center '>
						<p className='font-regular text-[15px] text-[#48525B]'>
							Already have an Account?
						</p>
						<button
							onClick={toggleContent}
							className='text-button ml-[5px] text-[15px] font-semibold text-[#FF8216] hover:text-blue-700 focus:outline-none'>
							Sign In
						</button>
					</div>
				</div>
			);
		} else {
			return (
				<div className='order-2 flex w-full flex-col items-center justify-center bg-sidebar py-[15px] sm:w-[50%] md:order-2 md:w-[50%] md:shadow-md '>
					{/* Logo Image */}
					<div>
						<img src={Pic} alt='My Image' />
					</div>
					<p className='mt-[5px] text-[20px] font-semibold text-[black]'>
						Sign in to
						<span className='text-[#FF8216]'>NewsWave</span>
					</p>

					{/* Email */}
					<div className='mt-[40px] flex w-[60%] justify-center '>
						<input
							type='text'
							className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
							placeholder='Username'
							value={username}
							onChange={e => setUsername(e.target.value)}
						/>
					</div>

					{/* Password */}
					<div className='mt-[12px] flex w-[60%]  justify-center '>
						<div className='relative w-full'>
							<input
								type={showPassword ? 'text' : 'password'}
								className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
								placeholder='Password'
								value={password}
								onChange={e => setPassword(e.target.value)}
								hidden={false}
							/>
							<div
								className='absolute right-0 top-0 flex h-full cursor-pointer items-center justify-center pr-[22px]'
								onClick={() => setShowPassword(!showPassword)}>
								{!showPassword ? (
									<svg
										width='28'
										height='18'
										viewBox='0 0 28 18'
										fill='none'
										xmlns='http://www.w3.org/2000/svg'>
										<path
											d='M27.83 8.45C20.1374 -2.53655 7.82641 -2.493 0.169959 8.45025C0.0631809 8.61337 0.00547689 8.80371 0.00370422 8.99866C0.00193154 9.19361 0.0561647 9.38497 0.159959 9.55C7.83312 20.5365 20.2485 20.5038 27.84 9.53974C27.9437 9.37653 27.9979 9.18677 27.9962 8.99339C27.9944 8.80001 27.9367 8.61128 27.83 8.45ZM14 14.44C12.5576 14.4385 11.1748 13.8649 10.155 12.845C9.13509 11.8251 8.56146 10.4423 8.55996 9C8.81487 1.8 19.1861 1.80186 19.44 9.00009C19.4384 10.4424 18.8648 11.8252 17.8449 12.845C16.8251 13.8649 15.4423 14.4385 14 14.44Z'
											fill='#FF8216'
										/>
										<path
											d='M14 5.56C13.0929 5.56789 12.2255 5.9338 11.5869 6.57808C10.9482 7.22235 10.5898 8.09283 10.5898 9.00003C10.5899 9.90723 10.9482 10.7777 11.5869 11.422C12.2256 12.0662 13.0929 12.4321 14.0001 12.44C14.9073 12.4321 15.7746 12.0662 16.4133 11.4219C17.052 10.7776 17.4103 9.90716 17.4103 8.99996C17.4103 8.09276 17.0519 7.22229 16.4132 6.57803C15.7745 5.93377 14.9072 5.56788 14 5.56Z'
											fill='#FF8216'
										/>
									</svg>
								) : (
									<svg
										width='30'
										height='22'
										viewBox='0 0 30 22'
										fill='none'
										xmlns='http://www.w3.org/2000/svg'>
										<path
											fillRule='evenodd'
											clipRule='evenodd'
											d='M24.2928 0.2925L4.34577 20.2395C3.95577 20.6295 3.95577 21.2625 4.34577 21.6535C4.73677 22.0435 5.36977 22.0435 5.75977 21.6535L25.7068 1.7065C26.0968 1.3165 26.0968 0.6825 25.7068 0.2925C25.3168 -0.0975 24.6828 -0.0975 24.2928 0.2925ZM23.8578 4.9695L19.6548 9.1725C19.8778 9.7385 19.9998 10.3545 19.9998 10.9995C19.9998 13.7595 17.7598 15.9995 14.9998 15.9995C14.3548 15.9995 13.7388 15.8775 13.1728 15.6545L10.2148 18.6125C11.7168 19.1575 13.3268 19.4995 14.9998 19.4995C23.1418 19.4995 29.7768 11.3925 29.7768 11.3925C30.0738 11.0255 30.0738 10.5005 29.7768 10.1335C29.7768 10.1335 27.4478 7.2885 23.8578 4.9695ZM6.43076 16.7405L10.3448 12.8265C10.1218 12.2605 9.99976 11.6445 9.99976 10.9995C9.99976 8.2395 12.2398 5.9995 14.9998 5.9995C15.6448 5.9995 16.2608 6.1215 16.8268 6.3445L20.1288 3.0425C18.5298 2.4235 16.8018 2.02649 14.9998 2.02649C6.85776 2.02649 0.22275 10.1335 0.22275 10.1335C-0.07425 10.5005 -0.07425 11.0255 0.22275 11.3925C0.22275 11.3925 2.67676 14.3905 6.43076 16.7405ZM17.9948 10.8325C17.9978 10.8875 17.9998 10.9435 17.9998 10.9995C17.9998 12.6555 16.6558 13.9995 14.9998 13.9995C14.9438 13.9995 14.8878 13.9975 14.8328 13.9945L17.9948 10.8325ZM12.0048 11.1665C12.0018 11.1115 11.9998 11.0555 11.9998 10.9995C11.9998 9.3435 13.3438 7.9995 14.9998 7.9995C15.0558 7.9995 15.1117 8.00151 15.1667 8.00451L12.0048 11.1665Z'
											fill='#FF8216'
										/>
									</svg>
								)}
							</div>
						</div>
					</div>

					{/* Remember me */}
					<div className='mt-[18px] flex w-[60%]  cursor-pointer select-none flex-row items-center gap-x-2 '>
						<div
							onClick={() => setChecked(!checked)}
							className='h-[17px] w-[17px] rounded-[4px] border-[2px] border-[#FF8216]'>
							{checked && (
								<div className='flex h-full w-full items-center justify-center bg-[#FF8216]'>
									<svg
										width='13'
										height='8'
										viewBox='0 0 13 8'
										fill='none'
										xmlns='http://www.w3.org/2000/svg'>
										<path
											d='M2.80984 6.74445C2.88584 6.86721 3.00861 6.96075 3.15476 6.98998C3.18984 6.99583 3.22491 7.00167 3.25999 7.00167C3.37107 7.00167 3.4763 6.9666 3.56399 6.90229L11.883 0.956819C12.1168 0.787282 12.1753 0.459902 12.0057 0.220212C11.8362 -0.0136315 11.5088 -0.0720928 11.2691 0.0974438L3.41199 5.70969L0.980016 1.68173C0.828018 1.43035 0.506482 1.35435 0.255101 1.5005C0.00371878 1.6525 -0.0722798 1.97404 0.0738724 2.22542L2.80984 6.74445Z'
											fill='white'
										/>
									</svg>
								</div>
							)}
						</div>
						<div>
							<p className='text-[#48525B]'>Remember Me</p>
						</div>
						<div className='ml-auto'>
							<p
								className='font-semibold text-[#FF8216]'
								onClick={handleModal}>
								Forgot Password
							</p>
						</div>
					</div>

					{/* Login Button */}
					<div className='mt-[30px] flex w-[60%]  justify-center'>
						<button
							onClick={handleLogin}
							className='h-[60px] w-full rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]'
							style={{ transition: '0.3s' }}>
							{loading ? (
								<span className='loading loading-spinner text-accent'></span>
							) : (
								<p>Log In</p>
							)}
						</button>
					</div>

					{/* SignUp Option */}
					<div className='mt-[20px] flex w-[60%]  justify-center'>
						<p className='font-regular text-[15px] text-[#48525B]'>
							Don't have an Account?
						</p>
						<button
							onClick={toggleContent}
							className='text-button ml-[5px] text-[15px] font-semibold text-[#FF8216] hover:text-blue-700 focus:outline-none'>
							Sign Up
						</button>
					</div>
				</div>
			);
		}
	};

	return (
		<div className='flex h-screen w-full flex-row flex-wrap'>
			{/* LOGO SPACE */}
			<div className='order-1 flex  w-full items-center justify-center bg-background sm:w-[50%] md:order-1 md:w-[50%]'>
				<div className='carousel h-full w-full'>
					<div id='slide1' className='carousel-item relative w-full'>
						{/* Image with overlay */}
						<div className=' carousel-item relative w-full'>
							<img
								src='https://images.unsplash.com/photo-1585829365295-ab7cd400c167?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80'
								className='w-full'
							/>
							{/* Overlay div */}
							<div className='absolute inset-0 bg-[#FF8216] bg-opacity-50'></div>
						</div>
						<div className='absolute left-5 right-5 top-1/2 flex -translate-y-1/2 transform justify-between'>
							<a href='#slide4' className='btn btn-circle'>
								❮
							</a>
							<a href='#slide2' className='btn btn-circle'>
								❯
							</a>
						</div>
					</div>
					<div id='slide2' className='carousel-item relative w-full'>
						{/* Image with overlay */}
						<div className=' carousel-item relative w-full'>
							<img
								src='https://images.unsplash.com/photo-1541560052-77ec1bbc09f7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1498&q=80'
								className='w-full'
							/>
							{/* Overlay div */}
							<div className='absolute inset-0 bg-[#FF8216] bg-opacity-50'></div>
						</div>
						<div className='absolute left-5 right-5 top-1/2 flex -translate-y-1/2 transform justify-between'>
							<a href='#slide1' className='btn btn-circle'>
								❮
							</a>
							<a href='#slide3' className='btn btn-circle'>
								❯
							</a>
						</div>
					</div>
					<div id='slide3' className='carousel-item relative w-full'>
						{/* Image with overlay */}
						<div className=' carousel-item relative w-full'>
							<img
								src='https://images.unsplash.com/photo-1543242594-c8bae8b9e708?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80'
								className='w-full'
							/>
							{/* Overlay div */}
							<div className='absolute inset-0 bg-[#FF8216] bg-opacity-50'></div>
						</div>
						<div className='absolute left-5 right-5 top-1/2 flex -translate-y-1/2 transform justify-between'>
							<a href='#slide2' className='btn btn-circle'>
								❮
							</a>
							<a href='#slide4' className='btn btn-circle'>
								❯
							</a>
						</div>
					</div>
					<div id='slide4' className='carousel-item relative w-full'>
						{/* Image with overlay */}
						<div className=' carousel-item relative w-full'>
							<img
								src='https://images.unsplash.com/photo-1603707676399-e143c4423f96?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1474&q=80'
								className='w-full'
							/>
							{/* Overlay div */}
							<div className='absolute inset-0 bg-[#FF8216] bg-opacity-50'></div>
						</div>
						<div className='absolute left-5 right-5 top-1/2 flex -translate-y-1/2 transform justify-between'>
							<a href='#slide3' className='btn btn-circle'>
								❮
							</a>
							<a href='#slide1' className='btn btn-circle'>
								❯
							</a>
						</div>
					</div>
				</div>
			</div>

			{/* Side View */}
			{switchingView()}
			{/* You can open the modal using document.getElementById('ID').showModal() method */}
			<ForgotPasswordModal handleModal={handleModal} />

			<ToastContainer />
		</div>
	);
};

export default Login;
