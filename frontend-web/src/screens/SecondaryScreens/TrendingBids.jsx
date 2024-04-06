import TrendingBidCards from '../../Components/BidComp/TrendingBidCards';
import { useLocation } from 'react-router-dom';

const TrendingBids = () => {
	const { state } = useLocation();
	return (
		<div className='flex w-full flex-col'>
			{/* Top Area */}
			<div className='flex w-full flex-col justify-between'>
				<h1 className='text-[20px] font-bold'> Active Bids</h1>
				<p>Welcome to bidding Page</p>
			</div>

			{/* Cards Area */}
			<div className='my-[22px] flex w-full  flex-row flex-wrap gap-3'>
				<TrendingBidCards data={state} />
			</div>
		</div>
	);
};

export default TrendingBids;
