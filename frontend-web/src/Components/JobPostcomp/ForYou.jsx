import UserOne from '../../assets/img1.png';
import { useNavigate } from 'react-router-dom';
import { HOSTNAME } from '../../../Config';

const ForYou = ({ data }) => {
	const navigate = useNavigate();
	return (
		<div>
			{data.map(element => (
				<div className='my-[15px] flex  w-full flex-row flex-wrap  rounded-xl border-[2px]'>
					<div className=' mb-[10px]  flex h-[80px] w-full  flex-row items-center justify-between rounded-xl  bg-[white] p-4 '>
						<div className='avatar items-center'>
							<div className='w-[50px] rounded-full'>
								<img
									src={
										element.creator.picture !== null
											? `${HOSTNAME}users/picture?path=${element.creator.picture}`
											: UserOne
									}
								/>
							</div>
						</div>

						<div className='mr-[auto]  px-5'>
							<p className='text-[14px] font-bold text-[black] sm:text-[24px]'>
								{element.creator.username}
							</p>
						</div>

						<button
							className='ml-[auto] h-[22px] w-[80px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]   text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[150px]'
							style={{ transition: '0.3s' }}
							onClick={() =>
								navigate(`/JobDetails`, {
									state: { ...element, past: false },
								})
							}>
							Go
						</button>
					</div>

					<div className='mb-[20px]  flex w-full flex-row justify-between rounded-xl bg-[white] px-4'>
						<div>
							<p className='text-[14px] text-[black] '>
								{element.details}
							</p>
							<p className='text-[12px] text-[black]'>
								{element.expiry_date}
							</p>
						</div>
					</div>
				</div>
			))}
		</div>
	);
};

export default ForYou;
