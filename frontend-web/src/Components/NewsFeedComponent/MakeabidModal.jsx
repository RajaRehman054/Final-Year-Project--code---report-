import React, { useState } from 'react';
import { createBid } from '../../api/Api';
import { toast } from 'react-toastify';
import { useSelector } from 'react-redux';
import { useDispatch } from 'react-redux';
import { setUser } from '../../redux/userSlice.js';
import { getUser } from '../../api/Api.js';

const MakeabidModal = ({ showModal, id }) => {
	const dispatch = useDispatch();
	const [checked, setChecked] = useState(false);
	const { user } = useSelector(state => state.user);
	const [price, setPrice] = useState(1);
	const fetchUsers = async () => {
		const data = await getUser();
		dispatch(setUser(data.user));
	};
	const createBidHandler = async () => {
		let payload = {
			video: id,
			amount: price,
		};
		if (user.wallet < price) {
			showModal(true);
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
		await createBid(payload);
		await fetchUsers();
		showModal(true);
		toast.success('Bid created successfully.', {
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
						<p className=' w-[50%]  text-[20px] font-bold'>
							Leading Bid
						</p>

						<p className=' w-[50%] text-center text-[20px] font-bold'></p>
					</div>
					<div className='mt-[30px] flex w-full flex-row items-center'>
						<p className=' w-[50%] text-[20px] font-bold'>
							Your Max Bid
						</p>

						<input
							className='h-[60px] w-[50%] rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none'
							placeholder='$'
							value={price}
							onChange={e => setPrice(e.target.value)}
						/>
					</div>
					<div className='mt-[20px] flex w-full'>
						<button
							className='h-[60px] w-full rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]'
							style={{ transition: '0.3s' }}
							onClick={createBidHandler}>
							Place a Bid
						</button>
					</div>
				</div>
			</div>
		</dialog>
	);
};

export default MakeabidModal;
