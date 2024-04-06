import React, { useState } from 'react';
import { updateBid } from '../../api/Api';
import { toast } from 'react-toastify';
import { useSelector, useDispatch } from 'react-redux';
import { getUser } from '../../api/Api.js';
import { setUser } from '../../redux/userSlice.js';

const contentdetailsModal = props => {
	const dispatch = useDispatch();
	const { data, handleModal } = props;
	const [amount, setAmount] = useState(data.current_amount + 1);
	const { user } = useSelector(state => state.user);
	const fetchUsers = async () => {
		const data = await getUser();
		dispatch(setUser(data.user));
	};

	const handleBid = async () => {
		const res = await updateBid(data._id, { amount });
		if (res?.success2 === false) {
			handleModal(true);
			return toast.error(`Not enough balance!`, {
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
		if (res?.success === false) {
			handleModal(true);
			return toast.error('Higher bid already placed.', {
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
		await fetchUsers();
		handleModal(true);
		toast.success('Bid added successfully.', {
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

	return (
		<dialog id='my_modal_3' className='modal '>
			<div className='modal-box w-[400px] bg-white '>
				<form method='dialog'>
					{/* if there is a button in form, it will close the modal */}
					<button className='btn btn-circle btn-ghost btn-sm absolute right-2 top-2'>
						✕
					</button>
				</form>
				<div>
					<h3 className='text-center text-lg font-bold text-[#FF8216]'>
						Place a Bid
					</h3>

					<div className='mt-[30px] flex w-full flex-row items-center'>
						<p className=' w-[50%] text-[20px] font-bold'>
							Leading Bid
						</p>

						<p className=' w-[50%] text-center text-[20px] font-bold'>
							{data.current_amount}
						</p>
					</div>
					<div className='mt-[30px] flex w-full flex-row items-center'>
						<p className=' w-[50%] text-[20px] font-bold'>
							Your Max Bid
						</p>

						<input
							className='h-[60px] w-[50%] rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
							placeholder='$'
							value={amount}
							onChange={e => setAmount(e.target.value)}
						/>
					</div>
					<div className='mt-[20px] flex w-full'>
						<button
							className='h-[60px] w-full rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]'
							style={{ transition: '0.3s' }}
							onClick={handleBid}>
							Place a Bid
						</button>
					</div>
				</div>
			</div>
		</dialog>
	);
};

export default contentdetailsModal;
