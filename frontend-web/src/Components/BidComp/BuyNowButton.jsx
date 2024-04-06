import React from 'react';
import axiosClient from '../../../axiosClient';
import { useSelector } from 'react-redux';
import { HOSTNAME } from '../../../Config';

const BuyNowButton = ({ videoItem }) => {
	const { user } = useSelector(state => state.user);

	const handleCheckout = () => {
		axiosClient
			.post(`${HOSTNAME}payment/create-checkout-session`, {
				videoItem,
			})
			.then(response => {
				if (response.data.url) {
					window.location.href = response.data.url;
				}
			})
			.catch(err => console.log(err.message));
	};

	return (
		<button
			className='h-[22px] w-[100px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]   text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
			style={{ transition: '0.3s' }}
			onClick={() => handleCheckout()}>
			Buy now
		</button>
	);
};

export default BuyNowButton;
