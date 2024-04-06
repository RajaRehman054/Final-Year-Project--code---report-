import React from 'react';
import { useNavigate } from 'react-router-dom';
import { format } from 'timeago.js';
import { HOSTNAME } from '../../../Config';

const TrendingBidCards = props => {
	const navigate = useNavigate();
	const { data } = props;
	return (
		<>
			{data.length > 0 ? (
				data.map(element => (
					<div className='card w-full bg-white shadow-xl sm:w-[24%] '>
						<figure className='px-10 pt-10'>
							<img
								src={`${HOSTNAME}videos/thumbnails?path=${element.video.thumbnail}`}
								alt='Shoes'
								className='rounded-xl '
							/>
						</figure>
						<div className='card-body border-[black]'>
							<h2 className='card-title'>
								{element.video.description}
							</h2>
							<div className='flex flex-row justify-between'>
								<p className='text-[14px]'>Auction time</p>
								<p className='text-[14px]'>Current Bid</p>
							</div>
							<div className='flex flex-row justify-between'>
								<p>expires {format(element.expires)}</p>
								<p className='text-[#FF8216]'>
									{element.current_amount}
								</p>
							</div>
							<div className='card-actions'>
								<button
									className='btn w-full bg-[#FF8216] font-bold text-[white]'
									onClick={() =>
										navigate(`/contentDetails`, {
											state: element,
										})
									}>
									Place a Bid
								</button>
							</div>
						</div>
					</div>
				))
			) : (
				<p>No active bids currently.</p>
			)}
		</>
	);
};

export default TrendingBidCards;
