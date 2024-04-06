import React, { useState } from 'react';
import { createJob } from '../../api/Api';
import { toast } from 'react-toastify';

const CreateNewJobModal = ({ showModal }) => {
	const [checked, setChecked] = useState(false);
	const [details, setDetails] = useState('');
	const [date, setDate] = useState('');
	const [price, setPrice] = useState('');

	const validateDate = inputDate => {
		const dateRegex = /^(0?[1-9]|1[0-2])\/(0?[1-9]|1\d|2\d|3[01])\/\d{4}$/;
		return dateRegex.test(inputDate);
	};

	const isExpired = inputDate => {
		const currentDate = new Date();
		const inputDateParts = inputDate.split('/');
		const month = parseInt(inputDateParts[0]) - 1;
		const day = parseInt(inputDateParts[1]);
		const year = parseInt(inputDateParts[2]);
		const expirationDate = new Date(year, month, day);
		return expirationDate < currentDate;
	};

	const jobCreation = async () => {
		if (date === '' || price === '' || details === '') {
			return toast.warn('Please fill all the fields.', {
				position: 'bottom-right',
				autoClose: 1400,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		if (!checked) {
			return toast.warn('Please check our terms and Policy', {
				position: 'bottom-right',
				autoClose: 1400,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		let validDate = validateDate(date);
		if (!validDate) {
			return toast.warn('Date is in incorrect format(MM/DD/YYYY)', {
				position: 'bottom-right',
				autoClose: 1400,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		let expired = isExpired(date);
		if (expired) {
			return toast.warn('Date entered has passed, enter a future date.', {
				position: 'bottom-right',
				autoClose: 1400,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		await createJob({ price, details, expiry_date: date });
		showModal(true);
		toast.success('Job created successfully.', {
			position: 'bottom-right',
			autoClose: 1400,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
	};

	return (
		<dialog id='my_modal_1' className='modal '>
			<div className='modal-box w-[400px] bg-white '>
				<form method='dialog'>
					{/* if there is a button in form, it will close the modal */}
					<button
						className='btn btn-circle btn-ghost btn-sm absolute right-2 top-2'
						onClick={showModal}>
						✕
					</button>
				</form>
				<div>
					<h3 className='text-center text-lg font-bold text-[#FF8216]'>
						Create New Job
					</h3>

					<div className='mt-[30px] flex w-full'>
						<input
							className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
							placeholder='Add details'
							value={details}
							onChange={e => setDetails(e.target.value)}
						/>
					</div>
					<div className='mt-[30px] flex w-full'>
						<input
							className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
							placeholder='Add Expiry Date(MM/DD/YYYY)'
							value={date}
							onChange={e => setDate(e.target.value)}
						/>
					</div>
					<div className='mt-[30px] flex w-full'>
						<input
							className='h-[60px] w-full rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
							value={price}
							onChange={e => setPrice(e.target.value)}
							placeholder='Add Price'
						/>
					</div>
					<div className='mt-[30px] flex flex-row items-center justify-center gap-x-3'>
						<div
							onClick={() => setChecked(!checked)}
							className='h-[17px] w-[8%] rounded-[4px] border-[2px] border-[#FF8216] '>
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
						<p className='text-[#48525B]'>
							I have read and agreed to Newswave
							<span className='text-[#FF8216]'>
								Terms of use, Privacy policy,
							</span>
							and
							<span className='text-[#FF8216]'>Disclaimer</span>
						</p>
					</div>
					<div className='mt-[20px] flex w-full'>
						<button
							className='h-[60px] w-full rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]'
							style={{ transition: '0.3s' }}
							onClick={jobCreation}>
							Post Job
						</button>
					</div>
				</div>
			</div>
		</dialog>
	);
};

export default CreateNewJobModal;
