import ActiveBidsTable from '../../Components/BidComp/ActiveBidsTable';
import { useLocation } from 'react-router-dom';

const ActiveBids = () => {
	const { state } = useLocation();
	return (
		<div className='flex w-full flex-col'>
			{/* Top Area */}
			<div className='flex w-full flex-col justify-between'>
				<h1 className='text-[20px] font-bold'> Active Bids</h1>
				<p>Welcome to bidding Page</p>
			</div>

			<div className='my-[22px] w-[full] overflow-x-auto'>
				<ActiveBidsTable data={state} />
			</div>
		</div>
	);
};

export default ActiveBids;
