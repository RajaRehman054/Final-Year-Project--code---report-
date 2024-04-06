import React, { useState, useEffect } from 'react';
import { IoSearch } from 'react-icons/io5';
import Trending from '../../Components/NewsFeedComponent/Trending';
import { searchVideosApi, getVideosApi } from '../../api/Api';
import Loader from '../../Components/Loader/Loader';
import UserOne from '../../assets/Search.jpg';
import UserTwo from '../../assets/noresults.png';

const Search = () => {
	const [search, setSearch] = useState('');
	const [selectedType, setSelectedType] = useState('username');
	const [videos, setVideos] = useState([]);
	const [loading, setLoading] = useState(false);
	const [isTyping, setIsTyping] = useState(false);
	const [isSearchPressed, setIsSearchPressed] = useState(false);

	const fetchData = async () => {
		setLoading(true);
		setIsSearchPressed(true);
		const data1 = await getVideosApi();
		setVideos(data1);
		setLoading(false);
	};

	useEffect(() => {
		fetchData();
	}, []);

	const handleInputChange = e => {
		setSearch(e.target.value);
		setIsTyping(e.target.value !== '');
	};

	const handleSearchPress = async () => {
		setIsTyping(false);
		setIsSearchPressed(true);
		setLoading(true);
		const data1 = await searchVideosApi(selectedType, search);
		setVideos(data1);
		setLoading(false);
	};

	if (loading) {
		return <Loader />;
	}

	return (
		<div className='flex w-full flex-col'>
			{/*Top Heading */}
			<div className='flex w-full '>
				<h1 className='text-[20px] font-bold'>Search</h1>
			</div>

			{/*View Area divided in two */}
			<div className='flex flex-wrap justify-between'>
				{/*First Column */}
				<div className='w-full sm:w-[75%]'>
					<div className='relative my-[22px] flex w-full flex-row  rounded-xl bg-white p-4 shadow-xl'>
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
									What are you looking for?
								</p>
							</div>
							<div className='w-full flex flex-row items-center relative z-10'>
								<select
									className='h-[50px] w-[200px] bg-white rounded-tl-[50px] rounded-bl-[50px] text-center focus:outline-none'
									style={{
										boxShadow:
											'0px 4px 10px 0px rgba(0, 0, 0, 0.10)',
									}}
									value={selectedType}
									onChange={e =>
										setSelectedType(e.target.value)
									}>
									<option disabled selected>
										Category
									</option>
									<option value={'username'}>Name</option>
									<option value={'tags'}>Tag</option>
									<option value={'description'}>
										Description
									</option>
								</select>
								<label
									htmlFor='searchInput'
									className='h-[50px] w-full flex-row items-center gap-x-6 bg-white px-3 text-[#AEB6CF] focus:outline-none flex justify-start'
									style={{
										boxShadow:
											'0px 4px 10px 0px rgba(0, 0, 0, 0.10)',
										cursor: 'text',
									}}>
									<IoSearch size={24} color='#AEB6CF' />
									<input
										type='text'
										id='searchInput' // Add an id to the input
										value={search}
										onChange={handleInputChange}
										className='w-full bg-white text-[#868DA3] placeholder:text-[#AEB6CF] focus:outline-none'
										placeholder='Search'
									/>
								</label>

								<div className='flex cursor-pointer items-center  '>
									<button
										className='h-[50px] w-[100px] justify-center bg-[#FF8216] text-[15px] text-white rounded-tr-[50px] rounded-br-[50px] hover:bg-[#F7A541] active:bg-[#FFD583]  '
										style={{
											boxShadow:
												'0px 4px 10px 0px rgba(0, 0, 0, 0.10)',
											transition: '0.3s',
										}}
										onClick={handleSearchPress}>
										Search
									</button>
								</div>
							</div>
						</div>
					</div>

					{/*Search Results  */}

					{isTyping && (
						<div className='relative my-[22px] flex w-full flex-col items-center'>
							<div className='rounded-full items-center bg-white shadow-xl p-3 '>
								<span className='loading loading-infinity loading-lg text-[#FF8216]'></span>
							</div>
						</div>
					)}

					{/*Results div*/}
					{isSearchPressed && (
						<div className='relative my-[22px] flex w-full flex-col items-center'>
							{videos.length === 0 ? (
								<div
									className='bg-cover w-[400px] bg-center h-[300px] rounded-xl relative '
									style={{
										backgroundImage: `url(${UserTwo})`,
									}}></div>
							) : (
								<Trending videos={videos} />
							)}
						</div>
					)}
				</div>

				{/* Second Column View */}
				<div className='hidden w-full sm:block sm:w-[23%]'></div>
			</div>
		</div>
	);
};

export default Search;
