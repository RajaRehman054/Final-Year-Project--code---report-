import React, { useState, useEffect } from 'react';
import { IoWalletOutline } from 'react-icons/io5';
import PaymentModal from '../../Components/paymentComponent/paymentmodal';
import PaymentHistoryTable from '../../Components/paymentComponent/PaymentHistoryTable';
import { paymentHistory } from '../../api/Api';
import { useSelector } from 'react-redux';
import UserOne from '../../assets/jobpost.png';

const Payment = () => {
	const { user } = useSelector(state => state.user);
	const handleModal = async checker => {
		if (checker?.bubbles === true) {
			return document.getElementById('my_modal_3').showModal();
		}
		document.getElementById('my_modal_3').close();
	};
	const [paymentSet, setpaymentSet] = useState([]);
	const fetchData = async () => {
		const res = await paymentHistory();
		setpaymentSet(res);
	};
	useEffect(() => {
		fetchData();
	}, []);

	return (
		<div className='flex w-full flex-col'>
			{/*Top Heading */}
			<div className='flex w-full '>
				<h1 className='text-[20px] font-bold'>My Wallet</h1>
			</div>

			{/*View Area divided in two */}
			<div className='flex flex-wrap justify-between'>
				{/*First Column */}
				<div className='w-full sm:w-[75%]'>
					<div className='relative my-[22px] flex w-full flex-row  rounded-xl bg-white p-4 shadow-xl'>
						<div className=' w-full  bg-[#FF8216] flex flex-row justify-between items-center bg-opacity-30 p-4 rounded-xl'>
							<div>
								<div className='flex flex-row items-center'>
									<p className='text-[#FF8216] text-[25px] font-bold mr-[10px]'>
										{user.wallet}$
									</p>
									<IoWalletOutline
										size={24}
										color='#FF8216'
									/>
								</div>

								<p className='text-[#FF8216] text-[15px] font-bold'>
									current Balance
								</p>
							</div>
							<button
								className='h-[22px] w-[100px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]   text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
								style={{ transition: '0.3s' }}
								onClick={handleModal}>
								+ Add Money to wallet
							</button>
						</div>
					</div>
					<div className='my-[22px] flex flex-row items-center justify-between'>
						<h1 className='text-[20px] font-bold'>
							Payment History
						</h1>
					</div>
					{/* payment table*/}
					{paymentSet.length === 0 ? (
						<div className='relative my-[22px] flex w-full flex-row h-30 rounded-xl bg-white p-4 shadow-xl'>
							<div
								className=' w-full items-center p-4 rounded-xl relative'
								style={{
									backgroundImage: `url(${UserOne})`,
									backgroundSize: 'contain', // Ensure the background image covers the entire container
								}}>
								{/* Opacity overlay div */}
								<div className='absolute inset-0 bg-[black] bg-opacity-50 rounded-xl'></div>

								<div className='text-center relative z-10'>
									<p className='font-[semibold] text-[50px] text-[white] my-5'>
										No transactions made yet
									</p>
								</div>
							</div>
						</div>
					) : (
						<div className='my-[22px] w-[full] overflow-x-auto'>
							<PaymentHistoryTable data={paymentSet} />
						</div>
					)}
				</div>

				{/* Second Column View */}
				<div className='hidden w-full sm:block sm:w-[23%]'></div>
			</div>
			<PaymentModal showModal={handleModal} />
		</div>
	);
};

export default Payment;
